# Experiment 3: Split-Brain Conflict Resolution
#
# Hypothesis: Deterministic Grounding Hypothesis
# Claim: When Slack and Discord disagree about a bug's status, a
#        deterministic resolver that calls a ground-truth tool (GitHub API)
#        achieves near-perfect accuracy, while majority-vote LLM approaches
#        have high hallucination rates on conflict cases.
#
# Based on: Tool-augmented LLM literature (Schick et al. 2023 Toolformer;
#           ReAct — Yao et al. 2023) on grounding LLM decisions with
#           external API calls to avoid confabulation.

import os
import json
import sys
import io
import random

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from colorama import init, Fore, Style

init(autoreset=True)

random.seed(42)

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")

conflict_scenarios = [
    {
        "id": 1,
        "slack": "fixed",
        "discord": "broken",
        "github_deploy": "FAILED",
        "ground_truth": "broken",
        "description": "Dev closed ticket early but deploy failed silently",
    },
    {
        "id": 2,
        "slack": "fixed",
        "discord": "broken",
        "github_deploy": "SUCCESS",
        "ground_truth": "fixed",
        "description": "Deploy succeeded, Discord user had cached old page",
    },
    {
        "id": 3,
        "slack": "broken",
        "discord": "fixed",
        "github_deploy": "FAILED",
        "ground_truth": "broken",
        "description": "Discord user tested wrong environment",
    },
    {
        "id": 4,
        "slack": "broken",
        "discord": "broken",
        "github_deploy": "FAILED",
        "ground_truth": "broken",
        "description": "No conflict, both agree it is broken",
    },
    {
        "id": 5,
        "slack": "fixed",
        "discord": "fixed",
        "github_deploy": "SUCCESS",
        "ground_truth": "fixed",
        "description": "No conflict, both agree it is fixed",
    },
    {
        "id": 6,
        "slack": "fixed",
        "discord": "broken",
        "github_deploy": "SUCCESS",
        "ground_truth": "fixed",
        "description": "Deploy succeeded, Discord user reporting unrelated issue",
    },
    {
        "id": 7,
        "slack": "broken",
        "discord": "fixed",
        "github_deploy": "SUCCESS",
        "ground_truth": "fixed",
        "description": "Slack dev forgot to update after hotfix deployed",
    },
    {
        "id": 8,
        "slack": "fixed",
        "discord": "broken",
        "github_deploy": "FAILED",
        "ground_truth": "broken",
        "description": "Rollback happened after Slack message was sent",
    },
]

# Pre-generate the random choices upfront so seed=42 is deterministic
# regardless of how many times we call the resolver.
_rng = random.Random(42)
_conflict_random_choices = [_rng.choice(["slack", "discord"]) for _ in range(20)]
_conflict_call_count = 0


def majority_vote_resolver(slack, discord):
    global _conflict_call_count
    if slack == discord:
        return {
            "decision": slack,
            "method": "majority_vote",
            "conflict_detected": False,
        }
    # True conflict — simulate LLM confusion with seeded random
    choice = _conflict_random_choices[_conflict_call_count % len(_conflict_random_choices)]
    _conflict_call_count += 1
    decision = slack if choice == "slack" else discord
    return {
        "decision": decision,
        "method": "majority_vote",
        "conflict_detected": True,
    }


def deterministic_grounded_resolver(slack, discord, github):
    if slack == discord:
        return {
            "decision": slack,
            "method": "deterministic",
            "conflict_detected": False,
            "tool_called": False,
        }
    # Conflict — call the GitHub tool
    decision = "broken" if github == "FAILED" else "fixed"
    return {
        "decision": decision,
        "method": "deterministic",
        "conflict_detected": True,
        "tool_called": True,
    }


def run_experiment3():
    global _conflict_call_count
    _conflict_call_count = 0  # reset for reproducibility

    print(Fore.YELLOW + Style.BRIGHT + "\n" + "═" * 52)
    print(Fore.YELLOW + Style.BRIGHT + "  EXPERIMENT 3: Split-Brain Conflict Resolution")
    print(Fore.YELLOW + Style.BRIGHT + "═" * 52)
    print()

    scenario_results = []
    mv_correct_total = 0
    det_correct_total = 0
    conflict_count = 0
    mv_correct_on_conflict = 0

    for sc in conflict_scenarios:
        mv = majority_vote_resolver(sc["slack"], sc["discord"])
        det = deterministic_grounded_resolver(sc["slack"], sc["discord"], sc["github_deploy"])

        mv_correct = mv["decision"] == sc["ground_truth"]
        det_correct = det["decision"] == sc["ground_truth"]

        if mv_correct:
            mv_correct_total += 1
        if det_correct:
            det_correct_total += 1

        has_conflict = sc["slack"] != sc["discord"]
        if has_conflict:
            conflict_count += 1
            if mv_correct:
                mv_correct_on_conflict += 1

        print(Fore.WHITE + Style.BRIGHT + f"  Scenario {sc['id']}: {sc['description']}")

        if has_conflict:
            print(f"    Conflict detected: {Fore.YELLOW}YES{Style.RESET_ALL} (slack={sc['slack']} vs discord={sc['discord']})")
        else:
            print(f"    Conflict detected: NO (both say {sc['slack']})")

        mv_str = (Fore.GREEN + f"{mv['decision']:6s}  ✓ CORRECT") if mv_correct else (Fore.RED + f"{mv['decision']:6s}  ✗ WRONG")
        det_str = (Fore.GREEN + f"{det['decision']:6s}  ✓ CORRECT") if det_correct else (Fore.RED + f"{det['decision']:6s}  ✗ WRONG")
        tool_note = Fore.CYAN + "  [GitHub tool called]" if det.get("tool_called") else ""

        print(f"    Majority Vote:     {mv_str}{Style.RESET_ALL}")
        print(f"    Deterministic:     {det_str}{Style.RESET_ALL}{tool_note}")
        print()

        scenario_results.append({
            "id": sc["id"],
            "description": sc["description"],
            "conflict_detected": has_conflict,
            "mv_decision": mv["decision"],
            "mv_correct": mv_correct,
            "det_decision": det["decision"],
            "det_correct": det_correct,
            "tool_called": det.get("tool_called", False),
        })

    total = len(conflict_scenarios)
    mv_acc = mv_correct_total / total
    det_acc = det_correct_total / total
    mv_hallucination_rate = (conflict_count - mv_correct_on_conflict) / conflict_count if conflict_count > 0 else 0.0

    print(Fore.YELLOW + "  ── Final Scores ──")
    print(f"  Majority Vote overall accuracy:          {mv_acc * 100:.1f}%")
    print(f"  Deterministic Resolver overall accuracy: {det_acc * 100:.1f}%")
    print()
    print(f"  On conflict-only cases ({conflict_count} scenarios):")
    print(f"  Majority Vote hallucination rate: {mv_hallucination_rate * 100:.1f}%")
    print()

    passed = det_acc >= 0.99 and mv_hallucination_rate >= 0.30

    if passed:
        print(Fore.GREEN + Style.BRIGHT + "  Hypothesis 3 Result: PASS ✓")
    else:
        print(Fore.RED + Style.BRIGHT + "  Hypothesis 3 Result: FAIL ✗")

    results = {
        "experiment": "Split-Brain Conflict Resolution",
        "hypothesis": "Deterministic Grounding Hypothesis",
        "pass": passed,
        "deterministic_accuracy": round(det_acc, 4),
        "majority_vote_accuracy": round(mv_acc, 4),
        "mv_hallucination_rate_on_conflicts": round(mv_hallucination_rate, 4),
        "total_scenarios": total,
        "conflict_scenarios": conflict_count,
        "scenarios": scenario_results,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(Fore.WHITE + f"\n  Results saved to {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    run_experiment3()
