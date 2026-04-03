# Experiment 1: Cross-Channel Coreference Linking
#
# Hypothesis: Blackboard Coreference Hypothesis
# Claim: Semantic embeddings can correctly link the same bug across
#        different communication channels (Discord, Email, Slack)
#        even when wording differs completely, outperforming simple
#        token-overlap baselines.
#
# Based on: Zhukova et al. (2026) NewsWCL50r — cross-document
#           coreference resolution using contextual embeddings.

import os
import json
import sys
import io
import itertools

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
from colorama import init, Fore, Style

init(autoreset=True)

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")

messages = {
    "discord": "NullPointerException at auth_service.py line 142, uid=None",
    "email": "Users cannot log in at all since this morning, totally broken",
    "slack": "Reverted PR #4421 — broke the authentication pipeline",
    "jira_unrelated": "UI misalignment on the dashboard widget for Safari v17",
}

ground_truth = {
    "discord": "BUG_A",
    "email": "BUG_A",
    "slack": "BUG_A",
    "jira_unrelated": "BUG_B",
}


def _tokenize(text):
    return set(text.lower().split())


def _same_bug_gt(ch1, ch2):
    return ground_truth[ch1] == ground_truth[ch2]


def baseline_token_overlap(messages, ground_truth, threshold=0.15):
    channels = list(messages.keys())
    pairs = list(itertools.combinations(channels, 2))
    results = []
    correct = 0

    for ch1, ch2 in pairs:
        t1 = _tokenize(messages[ch1])
        t2 = _tokenize(messages[ch2])
        intersection = t1 & t2
        union = t1 | t2
        score = len(intersection) / len(union) if union else 0.0
        predicted_same = score >= threshold
        actually_same = _same_bug_gt(ch1, ch2)
        is_correct = predicted_same == actually_same
        if is_correct:
            correct += 1
        results.append({
            "pair": f"{ch1}↔{ch2}",
            "score": round(score, 4),
            "predicted_same": predicted_same,
            "actually_same": actually_same,
            "correct": is_correct,
        })

    accuracy = correct / len(pairs)
    return accuracy, results


def semantic_blackboard_linker(messages, ground_truth, threshold=0.20):
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    print(Fore.YELLOW + "  Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    channels = list(messages.keys())
    texts = [messages[ch] for ch in channels]
    embeddings = model.encode(texts)

    pairs = list(itertools.combinations(range(len(channels)), 2))
    results = []
    correct = 0

    for i, j in pairs:
        ch1, ch2 = channels[i], channels[j]
        sim = float(cosine_similarity([embeddings[i]], [embeddings[j]])[0][0])
        predicted_same = sim >= threshold
        actually_same = _same_bug_gt(ch1, ch2)
        is_correct = predicted_same == actually_same
        if is_correct:
            correct += 1
        results.append({
            "pair": f"{ch1}↔{ch2}",
            "score": round(sim, 4),
            "predicted_same": predicted_same,
            "actually_same": actually_same,
            "correct": is_correct,
        })

    accuracy = correct / len(pairs)
    return accuracy, results


def run_experiment1():
    print(Fore.YELLOW + Style.BRIGHT + "\n" + "═" * 52)
    print(Fore.YELLOW + Style.BRIGHT + "  EXPERIMENT 1: Cross-Channel Coreference Linking")
    print(Fore.YELLOW + Style.BRIGHT + "═" * 52)

    baseline_acc, baseline_results = baseline_token_overlap(messages, ground_truth)
    semantic_acc, semantic_results = semantic_blackboard_linker(messages, ground_truth)

    # Build a lookup for semantic results by pair name
    semantic_lookup = {r["pair"]: r for r in semantic_results}

    print()
    for br in baseline_results:
        pair = br["pair"]
        sr = semantic_lookup[pair]

        print(Fore.WHITE + Style.BRIGHT + f"  Pair: {pair}")

        b_label = Fore.GREEN + "CORRECT ✓" if br["correct"] else Fore.RED + "WRONG ✗"
        s_label = Fore.GREEN + "CORRECT ✓" if sr["correct"] else Fore.RED + "WRONG ✗"

        print(f"    Baseline (token overlap): {b_label}{Style.RESET_ALL}  (score: {br['score']:.2f})")
        print(f"    Semantic (embedding):     {s_label}{Style.RESET_ALL}  (score: {sr['score']:.2f})")
        print()

    print(Fore.YELLOW + "  ── Final Scores ──")
    print(f"  Baseline Accuracy:  {baseline_acc * 100:.1f}%")
    print(f"  Semantic Accuracy:  {semantic_acc * 100:.1f}%")
    print()

    passed = semantic_acc >= 0.85
    if passed:
        print(Fore.GREEN + Style.BRIGHT + "  Hypothesis 1 Result: PASS ✓  (threshold was 85%)")
    else:
        print(Fore.RED + Style.BRIGHT + "  Hypothesis 1 Result: FAIL ✗  (threshold was 85%)")

    # Build pairs list for JSON (aligned with schema)
    pairs_json = []
    for br in baseline_results:
        pair = br["pair"]
        sr = semantic_lookup[pair]
        pairs_json.append({
            "pair": pair,
            "baseline_score": br["score"],
            "baseline_correct": br["correct"],
            "semantic_score": sr["score"],
            "semantic_correct": sr["correct"],
        })

    results = {
        "experiment": "Coreference Linking",
        "hypothesis": "Blackboard Coreference Hypothesis",
        "pass": passed,
        "key_metric": round(semantic_acc, 4),
        "key_metric_label": "Semantic Accuracy",
        "baseline_metric": round(baseline_acc, 4),
        "baseline_metric_label": "Token Overlap Accuracy",
        "pairs": pairs_json,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(Fore.WHITE + f"\n  Results saved to {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    run_experiment1()
