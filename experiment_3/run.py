# Experiment 3: Split-Brain Conflict Resolution
#
# Hypothesis: Deterministic Grounding Hypothesis
# Claim: When Slack and Discord disagree about a bug's status, a
#        deterministic resolver that calls a ground-truth tool (GitHub API)
#        achieves near-perfect accuracy, while an LLM given only the two
#        conflicting messages has a high error rate on conflict cases.
#
# Based on: Tool-augmented LLM literature (Schick et al. 2023 Toolformer;
#           ReAct — Yao et al. 2023) on grounding LLM decisions with
#           external API calls to avoid confabulation.

import os
import json
import sys
import io

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from colorama import init, Fore, Style

init(autoreset=True)
load_dotenv()

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configure Groq if key is available
_groq_client = None
if GROQ_API_KEY:
    try:
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_API_KEY)
        print(Fore.GREEN + "  Groq API configured successfully.")
    except Exception as e:
        print(Fore.YELLOW + f"  Warning: Groq setup failed ({e}). Will use fallback.")
else:
    print(Fore.YELLOW + "  Warning: GROQ_API_KEY not set. Using random fallback for majority vote.")

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


def _call_groq(slack, discord):
    """Call Groq with the two conflicting signals. Returns (decision, raw_response)."""
    prompt = (
        "You are a software incident triage assistant.\n\n"
        f'Slack reports: "{slack}"\n'
        f'Discord reports: "{discord}"\n\n'
        "These two channels disagree about whether a bug is currently fixed or broken.\n"
        "Based only on this information, what is your best assessment of the current status?\n\n"
        'Reply with exactly one word: either "fixed" or "broken".'
    )
    response = _groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    raw = response.choices[0].message.content.strip()
    decision = raw.lower().strip().rstrip(".")
    if decision not in ("fixed", "broken"):
        # If Groq didn't follow instructions, default to slack's value
        print(Fore.YELLOW + f"      [Groq parse warning: got '{raw}', defaulting to slack='{slack}']")
        decision = slack
    return decision, raw


def majority_vote_resolver(slack, discord):
    if slack == discord:
        return {
            "decision": slack,
            "method": "majority_vote",
            "conflict_detected": False,
            "groq_raw": None,
        }

    # True conflict — ask Groq (or fall back to random if no key)
    if _groq_client is not None:
        try:
            decision, raw = _call_groq(slack, discord)
            print(Fore.CYAN + f"      [Groq raw response: \"{raw}\"]")
            return {
                "decision": decision,
                "method": "majority_vote_groq",
                "conflict_detected": True,
                "groq_raw": raw,
            }
        except Exception as e:
            print(Fore.YELLOW + f"      [Groq call failed: {e}. Falling back to random.]")

    # Fallback: random with seed
    import random
    rng = random.Random(42)
    decision = rng.choice([slack, discord])
    return {
        "decision": decision,
        "method": "majority_vote_fallback",
        "conflict_detected": True,
        "groq_raw": None,
    }


def deterministic_grounded_resolver(slack, discord, github):
    if slack == discord:
        return {
            "decision": slack,
            "method": "deterministic",
            "conflict_detected": False,
            "tool_called": False,
        }
    decision = "broken" if github == "FAILED" else "fixed"
    return {
        "decision": decision,
        "method": "deterministic",
        "conflict_detected": True,
        "tool_called": True,
    }


def run_experiment3():
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

        mv_method_label = "Groq LLM" if "groq" in mv.get("method", "") else "Majority Vote"
        mv_str = (Fore.GREEN + f"{mv['decision']:6s}  ✓ CORRECT") if mv_correct else (Fore.RED + f"{mv['decision']:6s}  ✗ WRONG")
        det_str = (Fore.GREEN + f"{det['decision']:6s}  ✓ CORRECT") if det_correct else (Fore.RED + f"{det['decision']:6s}  ✗ WRONG")
        tool_note = Fore.CYAN + "  [GitHub tool called]" if det.get("tool_called") else ""

        print(f"    {mv_method_label}:       {mv_str}{Style.RESET_ALL}")
        print(f"    Deterministic:     {det_str}{Style.RESET_ALL}{tool_note}")
        print()

        scenario_results.append({
            "id": sc["id"],
            "description": sc["description"],
            "conflict_detected": has_conflict,
            "mv_decision": mv["decision"],
            "mv_correct": mv_correct,
            "mv_method": mv.get("method"),
            "groq_raw": mv.get("groq_raw"),
            "det_decision": det["decision"],
            "det_correct": det_correct,
            "tool_called": det.get("tool_called", False),
        })

    total = len(conflict_scenarios)
    mv_acc = mv_correct_total / total
    det_acc = det_correct_total / total
    mv_hallucination_rate = (conflict_count - mv_correct_on_conflict) / conflict_count if conflict_count > 0 else 0.0

    using_groq = _groq_client is not None
    method_label = "Groq LLM" if using_groq else "Majority Vote (fallback)"

    print(Fore.YELLOW + "  ── Final Scores ──")
    print(f"  {method_label} overall accuracy:    {mv_acc * 100:.1f}%")
    print(f"  Deterministic Resolver overall accuracy: {det_acc * 100:.1f}%")
    print()
    print(f"  On conflict-only cases ({conflict_count} scenarios):")
    print(f"  {method_label} error rate on conflicts: {mv_hallucination_rate * 100:.1f}%")
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
        "mv_method": "llama-3.3-70b-versatile" if using_groq else "random_fallback",
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
