#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import evaluate
import argparse
# For MoverScore
# import moverscore
# from moverscore import get_idf_dict, word_mover_score

# For Levenshtein
import Levenshtein

# For JS-Divergence
import numpy as np
from scipy.spatial import distance

def exact_match_score(prediction: str, reference: str) -> float:
    """Returns 1.0 if the trimmed prediction matches the trimmed reference exactly, else 0.0."""
    return 1.0 if prediction.strip() == reference.strip() else 0.0

def compute_moverscore(predictions, references):
    idf_refs = get_idf_dict(references)
    idf_hyps = get_idf_dict(predictions)
    scores = []
    for pred, ref in zip(predictions, references):
        s = word_mover_score([ref], pred, idf_refs, idf_hyps, stopwords=[])
        scores.append(s)
    return scores

def compute_levenshtein_distance(predictions, references):
    dists = []
    for pred, ref in zip(predictions, references):
        dist = Levenshtein.distance(pred, ref)
        dists.append(dist)
    return dists

def js_divergence(p, q):
    m = 0.5*(p + q)
    return 0.5*distance.rel_entr(p, m).sum() + 0.5*distance.rel_entr(q, m).sum()

def get_bow_distribution(text, vocab):
    tokens = text.split()
    counts = [0]*len(vocab)
    for t in tokens:
        if t in vocab:
            counts[vocab.index(t)] += 1
    total = sum(counts)
    probs = [c/total if total>0 else 0 for c in counts]
    return np.array(probs)

def compute_js_divergences(predictions, references):
    # Build vocab
    all_texts = predictions + references
    vocab_set = set()
    for txt in all_texts:
        vocab_set.update(txt.split())
    vocab = list(vocab_set)
    # Compute
    divergences = []
    for pred, ref in zip(predictions, references):
        p = get_bow_distribution(pred, vocab)
        q = get_bow_distribution(ref, vocab)
        div = js_divergence(p, q)
        divergences.append(div)
    return divergences

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM answers against ground-truth using various metrics."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="law_qa_llamataiwan.json",
        help="Path to the JSON file containing QA data."
    )
    args = parser.parse_args()

    # 2. Load data
    data_file_path = args.data_path
    with open(data_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = [item["model_answer"] for item in data]
    references = [item["correct_answer"] for item in data]

    print("Number of QA items:", len(data))

    # Exact Match
    exact_matches = []
    for pred, ref in zip(predictions, references):
        score = exact_match_score(pred, ref)
        exact_matches.append(score)
    exact_match_accuracy = sum(exact_matches) / len(exact_matches)
    print("Exact Match Accuracy:", exact_match_accuracy)

    # ROUGE
    rouge_metric = evaluate.load("rouge")
    rouge_results = rouge_metric.compute(predictions=predictions, references=references)
    print("\nROUGE Results:")
    for k, v in rouge_results.items():
        print(f"  {k}: {v:.4f}")

    # BLEU
    references_for_bleu = [[r] for r in references]
    bleu_metric = evaluate.load("bleu")
    bleu_results = bleu_metric.compute(predictions=predictions, references=references_for_bleu)
    print("\nBLEU Results:")
    print(f"  BLEU Score: {bleu_results['bleu']:.4f}")
    print(f"  n-gram Precisions: {bleu_results['precisions']}")
    print(f"  Brevity Penalty: {bleu_results['brevity_penalty']:.4f}")
    print(f"  Length Ratio: {bleu_results['length_ratio']:.4f}")
    
    # MoverScore
    # mover_scores = compute_moverscore(predictions, references)
    # avg_mover = sum(mover_scores)/len(mover_scores)
    # print("Average MoverScore:", avg_mover)

    # Levenshtein Distance
    lev_scores = compute_levenshtein_distance(predictions, references)
    avg_lev = sum(lev_scores)/len(lev_scores)
    print("Average Levenshtein Distance:", avg_lev)

    # JS-Divergence
    js_scores = compute_js_divergences(predictions, references)
    avg_js = sum(js_scores)/len(js_scores)
    print("Average JS-Divergence:", avg_js)
    
    # BERTScore
    bertscore_metric = evaluate.load("bertscore")
    bertscore_results = bertscore_metric.compute(
        predictions=predictions,
        references=references,
        model_type="bert-base-uncased"  # or any other available model
    )
    # Typically, we look at the F1 values
    avg_precision = sum(bertscore_results['precision']) / len(bertscore_results['precision'])
    avg_recall = sum(bertscore_results['recall']) / len(bertscore_results['recall'])
    avg_f1 = sum(bertscore_results['f1']) / len(bertscore_results['f1'])

    print("\nBERTScore (using bert-base-uncased):")
    print(f"  Average Precision: {avg_precision:.4f}")
    print(f"  Average Recall:    {avg_recall:.4f}")
    print(f"  Average F1:        {avg_f1:.4f}")

if __name__ == "__main__":
    main()
