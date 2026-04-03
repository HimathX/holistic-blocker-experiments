# Holistic Blocker Understanding

> NLP entity resolution across fragmented communication channels

This project demonstrates a proof-of-concept AI copilot designed to aggregate, deduplicate, and resolve bug reports (blockers) across fragmented communication channels such as Slack, Discord, Email, Jira, and Twitter. 

## Overview

Modern engineering teams suffer from "split-brain" tracking where bugs are reported in many discontinuous places. This leads to duplicated efforts, miscommunication, and untracked edge cases. This system uses Large Language Models (LLMs) and semantic search to achieve holistic blocker understanding through three key experiments, ensuring that your team has a single source of truth for issue triage.

## Project Structure

```text
.
├── dashboard.py           # Interactive Streamlit dashboard to visualize results
├── run_all.py             # Script to run all 3 experiments sequentially (headless)
├── pyproject.toml         # Python project metadata and dependencies
├── uv.lock                # Locked dependency tree for uv manager
├── .env                   # Configuration for API keys (e.g., GROQ_API_KEY)
├── experiment_1/          # Exp 1: Blackboard Coreference Linking
│   ├── run.py             # Script running the experiment
│   └── results.json       # Cached output data
├── experiment_2/          # Exp 2: Two-Stage Context Paging
│   ├── run.py             
│   └── results.json       
└── experiment_3/          # Exp 3: Split-Brain Conflict Resolution
    ├── run.py             
    └── results.json       
```

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for fast Python dependency management.

### Installation

1. **Clone the repository and install dependencies using `uv`:**
   ```bash
   # If you need to initialize the environment:
   uv venv
   
   # Synchronize dependencies:
   uv sync
   ```

2. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add your API keys. The system natively supports LLM APIs like Groq and Gemini.
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Experiments Details

### 1. Blackboard Coreference Linking
Tests whether semantic embeddings can identify the exact same bug described differently across distinct channels. It evaluates the accuracy of vector-based semantic similarity scoring relative to simple baseline token overlaps.

### 2. Two-Stage Context Paging
Explores token budget compression. It uses vector search as a low-cost Stage 1 filter to narrow down candidate bugs (finding the sweet spot 'k') before querying an LLM for final assignment. This setup mimics a RAG (Retrieval-Augmented Generation) pipeline, cutting out context window waste while maintaining accuracy.

### 3. Split-Brain Conflict Resolution
Evaluates the resolution of conflicting bug statuses across channels. It pairs native LLM conflict resolution techniques against external tool usage (Deterministic Grounding) where the model grounds its truth directly from GitHub deployment logs using models like Groq's `llama-3.3-70b-versatile`.

## Usage

### Run Experiments (Headless)
Run the entire suite of scripts headless to re-evaluate hypotheses and generate local `results.json` files for each experiment:
```bash
uv run run_all.py
```

### Launch Interactive Dashboard
We feature a comprehensive interactive Streamlit dashboard built to visualize the experiment results in full detail.
```bash
uv run streamlit run dashboard.py
```
