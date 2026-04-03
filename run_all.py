import os
import json
import sys
import io

# Force UTF-8 output on Windows so box-drawing / Unicode chars render correctly
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from colorama import init, Fore, Style

init(autoreset=True)

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")


def print_banner():
    print(Fore.CYAN + Style.BRIGHT + """
╔══════════════════════════════════════════════════════════╗
║        HOLISTIC BLOCKER UNDERSTANDING                    ║
║        Logical AI Copilot — Take-home Assignment         ║
╚══════════════════════════════════════════════════════════╝
""")


def run_all():
    print_banner()

    sys.path.insert(0, ROOT)

    # ── Experiment 1 ──────────────────────────────────────────
    print(Fore.CYAN + "▶  Running Experiment 1...")
    from experiment_1.run import run_experiment1
    r1 = run_experiment1()

    # ── Experiment 2 ──────────────────────────────────────────
    print(Fore.CYAN + "\n▶  Running Experiment 2...")
    from experiment_2.run import run_experiment2
    r2 = run_experiment2()

    # ── Experiment 3 ──────────────────────────────────────────
    print(Fore.CYAN + "\n▶  Running Experiment 3...")
    from experiment_3.run import run_experiment3
    r3 = run_experiment3()

    # ── Final Summary ─────────────────────────────────────────
    e1_pass = r1.get("pass", False)
    e2_pass = r2.get("pass", False)
    e3_pass = r3.get("pass", False)

    e1_metric = f"{r1.get('key_metric', 0) * 100:.1f}%"
    e2_metric = f"k={r2.get('sweet_spot_k', '?')}"
    e3_metric = f"{r3.get('deterministic_accuracy', 0) * 100:.1f}%"

    def pass_str(p, metric):
        icon = "PASS ✓" if p else "FAIL ✗"
        color = Fore.GREEN if p else Fore.RED
        return color + f"{icon}  {metric}" + Style.RESET_ALL

    print(Fore.YELLOW + Style.BRIGHT + """
╔══════════════════════════════════════════════════════════╗
║      HOLISTIC BLOCKER UNDERSTANDING — FINAL RESULTS      ║
╠══════════════════════════════════════════════════════════╣""")
    print(f"║  Experiment 1 (Coreference Linking):    {pass_str(e1_pass, e1_metric)}")
    print(f"║  Experiment 2 (Context Paging):         {pass_str(e2_pass, e2_metric)}")
    print(f"║  Experiment 3 (Conflict Resolution):    {pass_str(e3_pass, e3_metric)}")
    print(Fore.YELLOW + Style.BRIGHT + "╚══════════════════════════════════════════════════════════╝")

    # ── Save summary.json ─────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = {
        "experiments": [
            {
                "id": 1,
                "name": "Coreference Linking",
                "hypothesis": r1.get("hypothesis"),
                "pass": e1_pass,
                "key_metric": r1.get("key_metric"),
                "key_metric_label": r1.get("key_metric_label"),
            },
            {
                "id": 2,
                "name": "Two-Stage Context Paging",
                "hypothesis": r2.get("hypothesis"),
                "pass": e2_pass,
                "sweet_spot_k": r2.get("sweet_spot_k"),
                "token_budget_at_sweet_spot": r2.get("token_budget_at_sweet_spot"),
            },
            {
                "id": 3,
                "name": "Split-Brain Conflict Resolution",
                "hypothesis": r3.get("hypothesis"),
                "pass": e3_pass,
                "deterministic_accuracy": r3.get("deterministic_accuracy"),
                "majority_vote_accuracy": r3.get("majority_vote_accuracy"),
            },
        ]
    }

    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(Fore.WHITE + f"\n  Summary saved to {summary_path}\n")


if __name__ == "__main__":
    run_all()
