# Experiment 2: Two-Stage Context Paging
#
# Hypothesis: Two-Stage Context Paging Hypothesis
# Claim: A cheap vector pre-filter (Stage 1) can narrow a large bug history
#        down to a small candidate set so the LLM (Stage 2) only reads a
#        fraction of the full token budget while maintaining correct recall.
#
# Based on: RAG retrieval literature on top-k candidate selection and
#           token-budget-aware context compression for LLM inference.

import os
import json
import sys
import io

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
from colorama import init, Fore, Style

init(autoreset=True)

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")

bug_history = [
    {"id": "BUG_000", "description": "Payment service timeout on checkout"},
    {"id": "BUG_001", "description": "Image upload fails for PNG files over 5MB"},
    {"id": "BUG_002", "description": "Search results not updating after filter change"},
    {"id": "BUG_003", "description": "Notification emails going to spam folder"},
    {"id": "BUG_004", "description": "Dark mode toggle not persisting after refresh"},
    {"id": "BUG_005", "description": "CSV export missing last column of data"},
    {"id": "BUG_006", "description": "Video player controls disappear on mobile"},
    {"id": "BUG_007", "description": "Authentication service crash, users cannot access accounts"},
    {"id": "BUG_008", "description": "Calendar invite timezone showing incorrectly"},
    {"id": "BUG_009", "description": "Drag and drop broken in Firefox browser"},
    {"id": "BUG_010", "description": "API rate limit errors appearing too early"},
    {"id": "BUG_011", "description": "Profile picture not updating after upload"},
    {"id": "BUG_012", "description": "Markdown rendering broken in comment section"},
    {"id": "BUG_013", "description": "Two factor auth codes not being sent via SMS"},
    {"id": "BUG_014", "description": "Report generation freezes on large datasets"},
    {"id": "BUG_015", "description": "Autocomplete suggestions showing deleted items"},
    {"id": "BUG_016", "description": "Webhook delivery failing with 502 error"},
    {"id": "BUG_017", "description": "Graph tooltips overlapping on small screens"},
    {"id": "BUG_018", "description": "Session expiry happening too quickly"},
    {"id": "BUG_019", "description": "Database migration script failing on Postgres 15"},
]

new_message = "The auth module keeps throwing errors and users are completely locked out"
true_match = "BUG_007"
FULL_BUDGET_TOKENS = 20 * 150  # 3000


def stage1_vector_filter(new_message, bug_history, top_k, model):
    from sklearn.metrics.pairwise import cosine_similarity

    descriptions = [b["description"] for b in bug_history]
    all_texts = [new_message] + descriptions
    embeddings = model.encode(all_texts)

    query_emb = embeddings[0:1]
    bug_embs = embeddings[1:]

    sims = cosine_similarity(query_emb, bug_embs)[0]

    ranked_indices = np.argsort(sims)[::-1][:top_k]
    candidates = []
    for idx in ranked_indices:
        candidates.append({
            "id": bug_history[idx]["id"],
            "description": bug_history[idx]["description"],
            "score": round(float(sims[idx]), 4),
        })

    return candidates


def stage2_mock_llm_decision(new_message, candidates, true_match):
    candidate_ids = [c["id"] for c in candidates]
    token_cost = len(candidates) * 150

    if true_match in candidate_ids:
        decision = true_match
        correct = True
    else:
        decision = candidate_ids[0] if candidate_ids else None
        correct = False

    return {
        "decision": decision,
        "correct": correct,
        "token_cost": token_cost,
    }


def run_experiment2():
    from sentence_transformers import SentenceTransformer

    print(Fore.YELLOW + Style.BRIGHT + "\n" + "═" * 52)
    print(Fore.YELLOW + Style.BRIGHT + "  EXPERIMENT 2: Two-Stage Context Paging")
    print(Fore.YELLOW + Style.BRIGHT + "═" * 52)

    print(Fore.YELLOW + "  Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"\n  Full history token budget: {FULL_BUDGET_TOKENS} tokens\n")

    # Print similarity scores for all bugs so we can see the ranking
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    all_texts = [new_message] + [b["description"] for b in bug_history]
    embs = model.encode(all_texts)
    sims = cos_sim(embs[0:1], embs[1:])[0]
    ranked = np.argsort(sims)[::-1]

    print(Fore.YELLOW + "  Full similarity ranking (all 20 bugs):")
    for rank, idx in enumerate(ranked):
        bug = bug_history[idx]
        marker = " ◄ TRUE MATCH" if bug["id"] == true_match else ""
        print(f"    #{rank+1:2d}  {bug['id']}  {sims[idx]:.4f}  {bug['description'][:50]}{Fore.GREEN + marker if marker else ''}")
    print()

    k_values = [1, 3, 5, 10, 15, 20]
    k_results = []
    sweet_spot_k = None
    token_budget_at_sweet_spot = None

    for k in k_values:
        candidates = stage1_vector_filter(new_message, bug_history, k, model)
        candidate_ids = [c["id"] for c in candidates]
        stage1_hit = true_match in candidate_ids

        llm = stage2_mock_llm_decision(new_message, candidates, true_match)
        budget_pct = round((llm["token_cost"] / FULL_BUDGET_TOKENS) * 100, 1)
        passed = llm["correct"] and budget_pct <= 10.0

        if sweet_spot_k is None and llm["correct"] and budget_pct <= 10.0:
            sweet_spot_k = k
            token_budget_at_sweet_spot = budget_pct

        hit_str = Fore.GREEN + "YES" if stage1_hit else Fore.RED + "NO "
        correct_str = Fore.GREEN + "YES" if llm["correct"] else Fore.RED + "NO "
        pass_str = Fore.GREEN + "PASS ✓" if passed else Fore.RED + "FAIL ✗"

        print(
            f"  k={k:<2}  │ Stage1 Hit: {hit_str}{Style.RESET_ALL}"
            f"  │ Correct: {correct_str}{Style.RESET_ALL}"
            f"  │ Budget used: {budget_pct:5.1f}%"
            f"  │ {pass_str}{Style.RESET_ALL}"
        )

        k_results.append({
            "k": k,
            "stage1_hit": stage1_hit,
            "correct": llm["correct"],
            "budget_pct": budget_pct,
            "pass": passed,
        })

    print()
    if sweet_spot_k is not None:
        print(Fore.GREEN + f"  Sweet spot found at k={sweet_spot_k}")
    else:
        print(Fore.RED + "  No sweet spot found within 10% budget")

    # Pass condition: a sweet spot was found within 10% token budget
    passed = sweet_spot_k is not None and (token_budget_at_sweet_spot or 0) <= 10.0

    if sweet_spot_k is not None and token_budget_at_sweet_spot is None:
        sweet = next(r for r in k_results if r["k"] == sweet_spot_k)
        token_budget_at_sweet_spot = sweet["budget_pct"]

    print()
    if passed:
        print(Fore.GREEN + Style.BRIGHT + "  Hypothesis 2 Result: PASS ✓")
    else:
        print(Fore.RED + Style.BRIGHT + "  Hypothesis 2 Result: FAIL ✗")

    results = {
        "experiment": "Two-Stage Context Paging",
        "hypothesis": "Two-Stage Context Paging Hypothesis",
        "pass": passed,
        "sweet_spot_k": sweet_spot_k,
        "token_budget_at_sweet_spot": token_budget_at_sweet_spot,
        "full_budget_tokens": FULL_BUDGET_TOKENS,
        "k_results": k_results,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(Fore.WHITE + f"\n  Results saved to {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    run_experiment2()
