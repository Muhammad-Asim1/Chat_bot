"""
Automated RAG evaluation using Ragas.

Runs the production RAG pipeline against eval questions,
collects generated answers + retrieved contexts,
computes RAG metrics (including nDCG for retrieval quality), logs latency,
and saves a local report.

Reproducibility:
- Seeds are set at module load time
- Temperature = 0 for deterministic LLM outputs
- All components (embeddings, reranker, search) are deterministic
"""

from __future__ import annotations
import argparse
import os
import sys
import random
import time
import math

# Add parent directory to path for imports (config, rag_chain, vectorstore)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import List, Set
from datasets import Dataset
from dotenv import load_dotenv

# Set seeds BEFORE importing ML libraries
from config import Config
SEED = Config.RANDOM_SEED
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from dataset_eval import questions, answers as ground_truth_answers, ground_truth_chunk_ids
from dataset_eval import abstention_questions, abstention_ground_truth
from rag_chain import generate_answer_with_citations

from vectorstore import jina_embeddings  # your vectorstore

from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()

print(f"🎲 Evaluation running with seed={SEED}, temperature={Config.GENERATION_TEMPERATURE}")

ragas_llm = LangchainLLMWrapper(
    ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
)
)

# Utility: strip citation footers
def extract_answer_text(response_text: str) -> str:
    if not response_text:
        return ""
    delimiter = "---"
    if delimiter in response_text:
        return response_text.split(delimiter, 1)[0].strip()
    return response_text.strip()


def calculate_dcg(relevances: List[int], k: int = None) -> float:
    """
    Calculate Discounted Cumulative Gain (DCG) at position k.
    
    DCG = sum_{i=1}^{k} (rel_i / log2(i + 1))
    
    Args:
        relevances: List of relevance scores (1 for relevant, 0 for not relevant)
        k: Number of positions to consider (None = all positions)
    
    Returns:
        DCG score
    """
    if k is not None:
        relevances = relevances[:k]
    
    dcg = 0.0
    for i, rel in enumerate(relevances, start=1):
        dcg += rel / math.log2(i + 1)
    
    return dcg


def calculate_ndcg(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int = None) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (nDCG) at position k.
    
    nDCG = DCG / IDCG
    
    Where IDCG is the ideal DCG (all relevant documents at top positions).
    
    Args:
        retrieved_ids: List of retrieved chunk point IDs in ranked order
        ground_truth_ids: Set of relevant chunk point IDs
        k: Number of positions to consider (None = all positions)
    
    Returns:
        nDCG score between 0 and 1 (1 = perfect ranking)
    """
    if not ground_truth_ids:
        # No ground truth = skip this question (return NaN)
        return float('nan')
    
    # Calculate relevance scores for retrieved documents
    relevances = [1 if rid in ground_truth_ids else 0 for rid in retrieved_ids]
    
    # Calculate DCG
    dcg = calculate_dcg(relevances, k)
    
    # Calculate IDCG (ideal case: all relevant docs at top)
    num_relevant = len(ground_truth_ids)
    ideal_relevances = [1] * min(num_relevant, len(retrieved_ids) if k is None else k)
    idcg = calculate_dcg(ideal_relevances, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def calculate_recall_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int = None) -> float:
    """
    Calculate Recall@k - proportion of relevant documents retrieved in top k.
    
    Args:
        retrieved_ids: List of retrieved chunk point IDs in ranked order
        ground_truth_ids: Set of relevant chunk point IDs
        k: Number of positions to consider (None = all positions)
    
    Returns:
        Recall score between 0 and 1
    """
    if not ground_truth_ids:
        return float('nan')
    
    if k is not None:
        retrieved_ids = retrieved_ids[:k]
    
    retrieved_set = set(retrieved_ids)
    hits = len(retrieved_set & ground_truth_ids)
    
    return hits / len(ground_truth_ids)


def calculate_ais_metrics(cited_ids: List[str], ground_truth_ids: Set[str]) -> tuple[float, float, float]:
    """
    Calculate Answer Information Source (AIS) metrics.
    
    AIS measures whether the answer cites the correct source chunks.
    
    Args:
        cited_ids: List of chunk point IDs that were actually cited in the answer
        ground_truth_ids: Set of chunk point IDs that should have been cited
    
    Returns:
        Tuple of (citation_precision, ais_recall, ais_f1)
        - Citation-Precision: Of what was cited, how much was correct?
        - AIS-Recall: Of what should have been cited, how much was cited?
        - AIS-F1: Harmonic mean of precision and recall
    """
    if not ground_truth_ids:
        # No ground truth = skip this question
        return float('nan'), float('nan'), float('nan')
    
    cited_set = set(cited_ids)
    hits = len(cited_set & ground_truth_ids)
    
    # Citation-Precision: hits / cited (avoid div by zero if nothing was cited)
    if cited_set:
        citation_precision = hits / len(cited_set)
    else:
        citation_precision = 0.0
    
    # AIS-Recall: hits / ground_truth
    ais_recall = hits / len(ground_truth_ids)
    
    # AIS-F1: harmonic mean
    if citation_precision + ais_recall > 0:
        ais_f1 = 2 * (citation_precision * ais_recall) / (citation_precision + ais_recall)
    else:
        ais_f1 = 0.0
    
    return citation_precision, ais_recall, ais_f1


def is_abstention(response_text: str, chunks_metadata: list = None) -> tuple[bool, str]:
    """
    Detect if a response is an abstention (system refused to answer).
    
    The RAG system abstains in two ways:
    1. PRE-GENERATION: Relevance score below threshold → empty chunks_metadata, specific message
    2. POST-GENERATION: LLM recognizes it can't answer → specific phrase in response
    
    Args:
        response_text: The response from the RAG system
        chunks_metadata: List of retrieved chunks (empty = pre-generation abstention)
    
    Returns:
        Tuple of (is_abstention: bool, abstention_type: str)
        abstention_type is one of: "pre_generation", "post_generation", "none"
    """
    # Check 1: Pre-generation abstention (empty chunks = score too low)
    if chunks_metadata is not None and len(chunks_metadata) == 0:
        return True, "pre_generation"
    
    if not response_text:
        return True, "pre_generation"
    
    # Check 2: Post-generation abstention (LLM refused to answer)
    # Exact phrase from prompt instruction
    llm_abstention_phrase = "I don't have enough information in the provided documents to answer that question"
    
    if llm_abstention_phrase.lower() in response_text.lower():
        return True, "post_generation"
    
    # Check 3: Other abstention indicators (variations the LLM might use)
    abstention_indicators = [
        # Direct refusals
        "cannot be determined from the provided",
        "not mentioned in the provided",
        "no information available",
        "the documents do not contain",
        "outside the scope of the available documents",
        # "Unfortunately" style refusals
        "unfortunately, the provided documents do not",
        "the provided documents do not mention",
        "do not mention",
        "does not mention",
        # "Not explicitly" style
        "not explicitly stated in the provided",
        "not explicitly mentioned",
        "is not explicitly stated",
        "not stated in the provided",
        # Other variations
        "no relevant information found",
        "cannot find information",
        "unable to find",
        "not covered in the documents",
        "not addressed in the provided",
        "the documents don't contain",
        "the documents don't mention",
        "not available in the provided",
        "i could not find",
        "there is no information",
    ]
    
    response_lower = response_text.lower()
    for phrase in abstention_indicators:
        if phrase in response_lower:
            return True, "post_generation"
    
    return False, "none"


def run_abstention_evaluation(limit: int | None = None, save_path: str = "abstention_eval_results.csv", delay: float = 0):
    """
    Evaluate abstention effectiveness.
    
    Tests whether the RAG system correctly abstains from answering
    questions that are outside the scope of the available documents.
    
    Args:
        limit: Max number of questions to evaluate
        save_path: Path to save CSV results
        delay: Seconds to wait between API calls (to avoid rate limits)
    
    Metrics:
    - Abstention Rate: % of questions where system abstained
    - Abstention Recall: Of questions that SHOULD abstain, how many DID abstain?
    - False Answer Rate: % of questions where system incorrectly provided an answer
    """
    eval_questions = abstention_questions if limit is None else abstention_questions[:limit]
    eval_ground_truth = abstention_ground_truth if limit is None else abstention_ground_truth[:limit]
    
    data = {
        "question": [],
        "response": [],
        "should_abstain": [],
        "did_abstain": [],
        "abstention_type": [],  # "pre_generation", "post_generation", or "none"
        "num_chunks": [],
        "correct": [],
        "latency_sec": [],
    }
    
    latencies = []
    total = len(eval_questions)
    print(f"🧪 Running abstention evaluation on {total} questions...\n")
    
    abstained_count = 0
    pre_gen_count = 0
    post_gen_count = 0
    correct_count = 0
    
    for idx, (question, should_abstain) in enumerate(zip(eval_questions, eval_ground_truth), start=1):
        print(f"[{idx}/{total}] Query: {question[:60]}...")
        start_time = time.time()
        result = generate_answer_with_citations(question)
        latency = time.time() - start_time
        latencies.append(latency)
        
        # Unpack
        if isinstance(result, tuple):
            response_text, chunks_metadata = result
        else:
            response_text, chunks_metadata = result, []
        
        # Check if system abstained (pass both response and chunks)
        did_abstain, abstention_type = is_abstention(response_text, chunks_metadata)
        is_correct = (did_abstain == should_abstain)
        
        if did_abstain:
            abstained_count += 1
            if abstention_type == "pre_generation":
                pre_gen_count += 1
                type_label = "PRE-GEN: score below threshold"
            else:
                post_gen_count += 1
                type_label = "POST-GEN: LLM refused"
            
            if is_correct:
                print(f"   ✅ ABSTAINED [{type_label}]")
            else:
                print(f"   ⚠️ ABSTAINED (incorrect) [{type_label}]")
        else:
            print(f"   ❌ ANSWERED (should have abstained) | Chunks: {len(chunks_metadata)}")
        
        if is_correct:
            correct_count += 1
        
        data["question"].append(question)
        data["response"].append(response_text[:200] + "..." if len(response_text) > 200 else response_text)
        data["should_abstain"].append(should_abstain)
        data["did_abstain"].append(did_abstain)
        data["abstention_type"].append(abstention_type)
        data["num_chunks"].append(len(chunks_metadata))
        data["correct"].append(is_correct)
        data["latency_sec"].append(latency)
        
        # Rate limit delay
        if delay > 0 and idx < total:
            print(f"   ⏳ Waiting {delay}s before next query...")
            time.sleep(delay)
    
    # Calculate metrics
    abstention_rate = abstained_count / total if total > 0 else 0
    
    # For this dataset, all questions SHOULD abstain
    should_abstain_count = sum(eval_ground_truth)
    abstention_recall = abstained_count / should_abstain_count if should_abstain_count > 0 else 0
    
    # False answer rate = questions answered when they should have abstained
    false_answer_count = sum(1 for s, d in zip(eval_ground_truth, data["did_abstain"]) if s and not d)
    false_answer_rate = false_answer_count / total if total > 0 else 0
    
    accuracy = correct_count / total if total > 0 else 0
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    print("\n" + "="*60)
    print("📊 ABSTENTION EVALUATION RESULTS")
    print("="*60)
    
    print(f"\n📈 Summary:")
    print(f"- Total Questions: {total}")
    print(f"- Should Abstain: {should_abstain_count}")
    print(f"- Actually Abstained: {abstained_count}")
    print(f"  └─ Pre-generation (score < threshold): {pre_gen_count}")
    print(f"  └─ Post-generation (LLM refused): {post_gen_count}")
    print(f"- Correctly Handled: {correct_count}")
    
    print(f"\n🎯 Metrics:")
    print(f"- Abstention Rate: {abstention_rate:.2%}")
    print(f"- Abstention Recall: {abstention_recall:.2%} (of those that should abstain, how many did?)")
    print(f"- False Answer Rate: {false_answer_rate:.2%} (incorrectly answered when should abstain)")
    print(f"- Accuracy: {accuracy:.2%}")
    if abstained_count > 0:
        print(f"- Pre-gen Abstention %: {pre_gen_count/abstained_count:.1%} of abstentions")
        print(f"- Post-gen Abstention %: {post_gen_count/abstained_count:.1%} of abstentions")
    
    print(f"\n⏱️ Latency:")
    latencies_np = np.array(latencies)
    print(f"- Mean: {latencies_np.mean():.3f}s")
    print(f"- P95: {np.percentile(latencies_np, 95):.3f}s")
    
    # Show failures (questions that should have abstained but didn't)
    failures = [(q, r) for q, r, s, d in zip(
        data["question"], data["response"], data["should_abstain"], data["did_abstain"]
    ) if s and not d]
    
    if failures:
        print(f"\n❌ Failed Abstentions ({len(failures)} questions answered incorrectly):")
        for q, r in failures[:5]:  # Show first 5
            print(f"   Q: {q[:50]}...")
            print(f"   A: {r[:100]}...")
            print()
    
    df.to_csv(save_path, index=False)
    print(f"\n💾 Saved abstention results to {save_path}")
    
    return {
        "abstention_rate": abstention_rate,
        "abstention_recall": abstention_recall,
        "false_answer_rate": false_answer_rate,
        "accuracy": accuracy,
    }


# Main evaluation loop - generation incremental, Ragas in batch
def run_evaluation(
    start_idx: int = 0,
    end_idx: int | None = None,
    save_path: str = "ragas_eval_results.csv",
    delay: float = 0,
    batch_size: int = 10,
    batch_delay: float = 5.0
):
    """
    Run main RAG evaluation with Ragas metrics.
    
    - Generation is done incrementally (saved to intermediate CSV for crash safety)
    - Ragas metrics are computed in batch at the end (faster)
    
    Args:
        start_idx: Starting question index (0-based)
        end_idx: Ending question index (exclusive, None = all remaining)
        save_path: Path to save final CSV results
        delay: Seconds to wait between API calls (to avoid rate limits)
    """
    total_questions = len(questions)
    
    # Validate indices
    if start_idx < 0:
        start_idx = 0
    if end_idx is None or end_idx > total_questions:
        end_idx = total_questions
    if start_idx >= end_idx:
        print(f"❌ Invalid range: start_idx={start_idx}, end_idx={end_idx}")
        return
    
    eval_questions = questions[start_idx:end_idx]
    eval_ground_truths = ground_truth_answers[start_idx:end_idx]
    eval_chunk_ids = ground_truth_chunk_ids[start_idx:end_idx]
    
    total = len(eval_questions)
    intermediate_path = save_path.replace('.csv', '_intermediate.csv')
    
    print(f"🧪 Running evaluation on questions {start_idx} to {end_idx-1} ({total} questions)...")
    print(f"💾 Intermediate results: {intermediate_path}")
    print(f"💾 Final results: {save_path}\n")
    
    # Data storage for batch Ragas
    all_data = {
        "idx": [],
        "question": [],
        "answer": [],
        "ground_truth": [],
        "contexts": [],
        "ndcg": [],
        "ndcg@3": [],
        "ndcg@5": [],
        "recall@3": [],
        "recall@5": [],
        "citation_precision": [],
        "ais_recall": [],
        "ais_f1": [],
        "latency_sec": [],
        "num_chunks": [],
        "num_cited": [],
    }
    
    # Check if intermediate file exists (for resume)
    file_exists = os.path.exists(intermediate_path)
    
    # ========== PHASE 1: Generate answers incrementally ==========
    print("=" * 60)
    print("📝 PHASE 1: Generating answers (incremental save)")
    print("=" * 60)
    
    for i, (question, ground_truth, gt_chunk_ids) in enumerate(
        zip(eval_questions, eval_ground_truths, eval_chunk_ids)
    ):
        global_idx = start_idx + i
        print(f"\n[{global_idx}/{total_questions-1}] Query: {question[:70]}...")
        
        # Generate answer
        print(f"   🔍 Generating answer...")
        start_time = time.time()
        result = generate_answer_with_citations(question)
        latency = time.time() - start_time
        
        # Unpack
        if isinstance(result, tuple):
            response_text, chunks_metadata = result
        else:
            response_text, chunks_metadata = result, []
        
        answer_text = extract_answer_text(response_text)
        chunk_texts = [chunk.get("document") or chunk.get("preview") or "" for chunk in chunks_metadata]
        if not chunk_texts:
            chunk_texts = [""]
        
        # Calculate retrieval metrics
        retrieved_ids = [chunk.get("point_id") for chunk in chunks_metadata if chunk.get("point_id")]
        gt_ids_set = set(gt_chunk_ids)
        cited_ids = [chunk.get("point_id") for chunk in chunks_metadata if chunk.get("cited") and chunk.get("point_id")]
        
        ndcg_full = calculate_ndcg(retrieved_ids, gt_ids_set)
        ndcg_3 = calculate_ndcg(retrieved_ids, gt_ids_set, k=3)
        ndcg_5 = calculate_ndcg(retrieved_ids, gt_ids_set, k=5)
        recall_3 = calculate_recall_at_k(retrieved_ids, gt_ids_set, k=3)
        recall_5 = calculate_recall_at_k(retrieved_ids, gt_ids_set, k=5)
        cite_prec, ais_rec, ais_f1 = calculate_ais_metrics(cited_ids, gt_ids_set)
        
        hits = len(set(retrieved_ids) & gt_ids_set)
        if gt_ids_set:
            print(f"   📊 nDCG: {ndcg_full:.3f} | nDCG@5: {ndcg_5:.3f} | Hits: {hits}/{len(gt_ids_set)}")
        
        # Store for batch Ragas
        all_data["idx"].append(global_idx)
        all_data["question"].append(question)
        all_data["answer"].append(answer_text)
        all_data["ground_truth"].append(ground_truth)
        all_data["contexts"].append(chunk_texts)
        all_data["ndcg"].append(ndcg_full)
        all_data["ndcg@3"].append(ndcg_3)
        all_data["ndcg@5"].append(ndcg_5)
        all_data["recall@3"].append(recall_3)
        all_data["recall@5"].append(recall_5)
        all_data["citation_precision"].append(cite_prec)
        all_data["ais_recall"].append(ais_rec)
        all_data["ais_f1"].append(ais_f1)
        all_data["latency_sec"].append(latency)
        all_data["num_chunks"].append(len(chunks_metadata))
        all_data["num_cited"].append(len(cited_ids))
        
        # Save intermediate (without Ragas metrics)
        row = {
            "idx": global_idx,
            "question": question,
            "answer": answer_text[:500],
            "ground_truth": ground_truth[:500],
            "ndcg": ndcg_full,
            "ndcg@3": ndcg_3,
            "ndcg@5": ndcg_5,
            "recall@3": recall_3,
            "recall@5": recall_5,
            "citation_precision": cite_prec,
            "ais_recall": ais_rec,
            "ais_f1": ais_f1,
            "latency_sec": latency,
            "num_chunks": len(chunks_metadata),
            "num_cited": len(cited_ids),
        }
        row_df = pd.DataFrame([row])
        row_df.to_csv(intermediate_path, mode='a', header=not file_exists, index=False)
        file_exists = True
        
        print(f"   💾 Saved intermediate")
        
        # Rate limit delay
        if delay > 0 and i < total - 1:
            print(f"   ⏳ Waiting {delay}s...")
            time.sleep(delay)
    
    # ========== PHASE 2: Batch Ragas evaluation with retry ==========
    print("\n" + "=" * 60)
    print("🧮 PHASE 2: Computing Ragas metrics (batched with retry)")
    print("=" * 60)
    print(f"   Batch size: {batch_size}, Batch delay: {batch_delay}s")
    
    num_batches = (total + batch_size - 1) // batch_size
    print(f"   Processing {total} questions in {num_batches} batches...")
    
    all_ragas_results = []
    
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        
        print(f"\n   📦 Batch {batch_idx + 1}/{num_batches} (questions {start}-{end-1})...")
        
        batch_result = run_ragas_batch_with_retry(
            questions=all_data["question"][start:end],
            answers=all_data["answer"][start:end],
            contexts=all_data["contexts"][start:end],
            ground_truths=all_data["ground_truth"][start:end],
            max_retries=3,
            retry_delay=30.0
        )
        
        all_ragas_results.append(batch_result)
        
        # Show batch summary
        faith_mean = batch_result['faithfulness'].mean()
        rel_mean = batch_result['answer_relevancy'].mean()
        print(f"      ✅ Batch done: faith={faith_mean:.3f}, rel={rel_mean:.3f}")
        
        # Delay between batches (except last)
        if batch_idx < num_batches - 1:
            print(f"      ⏳ Waiting {batch_delay}s before next batch...")
            time.sleep(batch_delay)
    
    # Combine all batch results
    ragas_df = pd.concat(all_ragas_results, ignore_index=True)
    print(f"\n   ✅ Ragas evaluation complete!")
    
    # ========== PHASE 3: Merge and save final results ==========
    print("\n" + "=" * 60)
    print("💾 PHASE 3: Saving final results")
    print("=" * 60)
    
    # Build final dataframe
    final_df = pd.DataFrame({
        "idx": all_data["idx"],
        "question": all_data["question"],
        "answer": [a[:500] for a in all_data["answer"]],
        "ground_truth": [g[:500] for g in all_data["ground_truth"]],
        "faithfulness": ragas_df["faithfulness"].tolist(),
        "answer_relevancy": ragas_df["answer_relevancy"].tolist(),
        "context_precision": ragas_df["context_precision"].tolist(),
        "context_recall": ragas_df["context_recall"].tolist(),
        "ndcg": all_data["ndcg"],
        "ndcg@3": all_data["ndcg@3"],
        "ndcg@5": all_data["ndcg@5"],
        "recall@3": all_data["recall@3"],
        "recall@5": all_data["recall@5"],
        "citation_precision": all_data["citation_precision"],
        "ais_recall": all_data["ais_recall"],
        "ais_f1": all_data["ais_f1"],
        "latency_sec": all_data["latency_sec"],
        "num_chunks": all_data["num_chunks"],
        "num_cited": all_data["num_cited"],
    })
    
    final_df.to_csv(save_path, index=False)
    print(f"   ✅ Saved to {save_path}")
    
    # ========== Summary ==========
    print("\n" + "=" * 60)
    print("📊 EVALUATION COMPLETE")
    print("=" * 60)
    
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall",
                   "ndcg", "ndcg@3", "ndcg@5", "recall@3", "recall@5",
                   "citation_precision", "ais_recall", "ais_f1"]
    
    print("\n🧮 Aggregate Scores:")
    for col in metric_cols:
        if col in final_df.columns:
            valid = final_df[col].dropna()
            if len(valid) > 0:
                print(f"  - {col}: {valid.mean():.4f}")
    
    print(f"\n⏱️ Latency: mean={final_df['latency_sec'].mean():.2f}s, max={final_df['latency_sec'].max():.2f}s")

def run_ragas_batch_with_retry(
    questions: list,
    answers: list, 
    contexts: list,
    ground_truths: list,
    max_retries: int = 3,
    retry_delay: float = 30.0
) -> pd.DataFrame:
    """
    Run Ragas evaluation on a batch with retry logic.
    
    Args:
        questions, answers, contexts, ground_truths: Data for Ragas
        max_retries: Number of retry attempts
        retry_delay: Seconds to wait between retries (doubles each retry)
    
    Returns:
        DataFrame with Ragas scores
    """
    ragas_metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    
    ragas_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truths": ground_truths,
        "reference": ground_truths,
    }
    dataset = Dataset.from_dict(ragas_data)
    
    for attempt in range(max_retries):
        try:
            result = evaluate(dataset=dataset, metrics=ragas_metrics, llm=ragas_llm, embeddings=jina_embeddings)
            return result.to_pandas()
        except Exception as e:
            wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
            if attempt < max_retries - 1:
                print(f"      ⚠️ Attempt {attempt + 1} failed: {str(e)[:50]}...")
                print(f"      ⏳ Waiting {wait_time:.0f}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"      ❌ All {max_retries} attempts failed: {e}")
                # Return NaN dataframe
                return pd.DataFrame({
                    "faithfulness": [float('nan')] * len(questions),
                    "answer_relevancy": [float('nan')] * len(questions),
                    "context_precision": [float('nan')] * len(questions),
                    "context_recall": [float('nan')] * len(questions),
                })


def run_ragas_only(
    intermediate_path: str, 
    save_path: str = "ragas_eval_results.csv",
    batch_size: int = 10,
    batch_delay: float = 5.0
):
    """
    Run only Ragas evaluation on an existing intermediate CSV file.
    
    Uses batching with retry logic to handle rate limits.
    
    Args:
        intermediate_path: Path to intermediate CSV with generated answers
        save_path: Path to save final results with Ragas metrics
        batch_size: Number of questions per Ragas batch (default 10)
        batch_delay: Seconds to wait between batches (default 5)
    """
    import ast
    
    print("=" * 60)
    print("🧮 RAGAS-ONLY EVALUATION (with batching & retry)")
    print("=" * 60)
    print(f"   Batch size: {batch_size}")
    print(f"   Batch delay: {batch_delay}s")
    
    # Read intermediate file
    print(f"\n📂 Reading: {intermediate_path}")
    if not os.path.exists(intermediate_path):
        print(f"❌ File not found: {intermediate_path}")
        return
    
    intermediate_df = pd.read_csv(intermediate_path)
    intermediate_df = intermediate_df.sort_values('idx').reset_index(drop=True)
    print(f"   Found {len(intermediate_df)} rows")
    print(f"   Columns: {list(intermediate_df.columns)}")
    
    # Check for contexts column
    if 'contexts' not in intermediate_df.columns:
        print("\n⚠️  WARNING: No 'contexts' column found!")
        print("   Ragas needs contexts to compute faithfulness, answer_relevancy, etc.")
        print("   Will use ground_truth as fallback context (results may be unreliable)")
        contexts_list = [[gt] for gt in intermediate_df['ground_truth'].tolist()]
    else:
        # Parse contexts from string
        def parse_contexts(ctx_str):
            try:
                return ast.literal_eval(ctx_str)
            except:
                return [""]
        contexts_list = [parse_contexts(c) for c in intermediate_df['contexts'].tolist()]
        print(f"   ✅ Contexts column found")
    
    # Prepare full data
    all_questions = intermediate_df['question'].tolist()
    all_answers = intermediate_df['answer'].tolist()
    all_ground_truths = intermediate_df['ground_truth'].tolist()
    
    total = len(all_questions)
    num_batches = (total + batch_size - 1) // batch_size
    
    print(f"\n🧮 Running Ragas in {num_batches} batches of {batch_size}...")
    
    # Collect all results
    all_ragas_results = []
    
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        
        print(f"\n   📦 Batch {batch_idx + 1}/{num_batches} (questions {start}-{end-1})...")
        
        batch_questions = all_questions[start:end]
        batch_answers = all_answers[start:end]
        batch_contexts = contexts_list[start:end]
        batch_ground_truths = all_ground_truths[start:end]
        
        # Run with retry
        batch_result = run_ragas_batch_with_retry(
            questions=batch_questions,
            answers=batch_answers,
            contexts=batch_contexts,
            ground_truths=batch_ground_truths,
            max_retries=3,
            retry_delay=30.0
        )
        
        all_ragas_results.append(batch_result)
        
        # Show batch summary
        faith_mean = batch_result['faithfulness'].mean()
        rel_mean = batch_result['answer_relevancy'].mean()
        print(f"      ✅ Batch done: faith={faith_mean:.3f}, rel={rel_mean:.3f}")
        
        # Delay between batches (except last)
        if batch_idx < num_batches - 1:
            print(f"      ⏳ Waiting {batch_delay}s before next batch...")
            time.sleep(batch_delay)
    
    # Combine all batch results
    ragas_df = pd.concat(all_ragas_results, ignore_index=True)
    
    # Merge with intermediate data
    print(f"\n💾 Saving to: {save_path}")
    final_df = intermediate_df.copy()
    final_df["faithfulness"] = ragas_df["faithfulness"].tolist()
    final_df["answer_relevancy"] = ragas_df["answer_relevancy"].tolist()
    final_df["context_precision"] = ragas_df["context_precision"].tolist()
    final_df["context_recall"] = ragas_df["context_recall"].tolist()
    
    # Reorder columns
    priority_cols = ["idx", "question", "answer", "ground_truth", 
                     "faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    other_cols = [c for c in final_df.columns if c not in priority_cols and c != 'contexts']
    final_df = final_df[priority_cols + other_cols]
    
    final_df.to_csv(save_path, index=False)
    print(f"   ✅ Saved {len(final_df)} rows")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 RESULTS SUMMARY")
    print("=" * 60)
    
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall",
                   "ndcg", "ndcg@3", "ndcg@5", "recall@3", "recall@5",
                   "citation_precision", "ais_recall", "ais_f1"]
    
    print("\n🧮 Aggregate Scores:")
    for col in metric_cols:
        if col in final_df.columns:
            valid = final_df[col].dropna()
            if len(valid) > 0:
                print(f"  - {col}: {valid.mean():.4f}")


# CLI
def parse_args():
    parser = argparse.ArgumentParser(description="Run Ragas evaluation with Gemini + Jina embeddings")
    parser.add_argument("--start", type=int, default=0, help="Starting question index (0-based)")
    parser.add_argument("--end", type=int, default=None, help="Ending question index (exclusive)")
    parser.add_argument("--save", type=str, default="ragas_eval_results.csv", help="Save CSV path")
    parser.add_argument("--abstention", action="store_true", help="Run abstention evaluation instead of main eval")
    parser.add_argument("--delay", type=float, default=0, help="Delay in seconds between API calls to avoid rate limits")
    parser.add_argument("--ragas-only", type=str, metavar="INTERMEDIATE_CSV", 
                        help="Run only Ragas evaluation on existing intermediate CSV file")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for Ragas evaluation (default 10)")
    parser.add_argument("--batch-delay", type=float, default=5.0, help="Delay between Ragas batches in seconds (default 5)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.ragas_only:
        run_ragas_only(
            intermediate_path=args.ragas_only, 
            save_path=args.save,
            batch_size=args.batch_size,
            batch_delay=args.batch_delay
        )
    elif args.abstention:
        save_path = args.save if args.save != "ragas_eval_results.csv" else "abstention_eval_results.csv"
        # For abstention, use start/end if provided
        limit = (args.end - args.start) if args.end else None
        run_abstention_evaluation(limit=limit, save_path=save_path, delay=args.delay)
    else:
        run_evaluation(
            start_idx=args.start, 
            end_idx=args.end, 
            save_path=args.save, 
            delay=args.delay,
            batch_size=args.batch_size,
            batch_delay=args.batch_delay
        )
