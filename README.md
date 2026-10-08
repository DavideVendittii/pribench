# PriBench: A Balanced Benchmark for Diagnosing Privacy Leakage Vulnerabilities in Large Language Models

This repository contains the official dataset, evaluation suite, and analysis tools for **PriBench**, a balanced benchmark designed to diagnose privacy leakage vulnerabilities in Large Language Models (LLMs) across direct and agentic delegation settings.

---

## 📌 Repository Structure

```text
pribench/
├── README.md
├── requirements.txt
├── data/
│   ├── pribench_dataset.csv              # Full dataset (6,284 balanced prompts)
│   └── in_context_disclosure_suite.csv   # Synthetic in-context evaluation suite (300 scenarios)
├── eval/
│   ├── model_vulnerability_eval.py        # Native baseline Leakage Vulnerability Rate (LVR)
│   ├── evaluate_framing_variations.py     # Delegation conditions (A–C) and Identity Framing
│   ├── evaluate_incontext_disclosure.py   # In-context access control and Actual Disclosure Rate (ADR)
│   └── evaluate_fine_grained_cross_judge.py # Peer Cross-Judge Evaluation suite
└── utils/
    ├── audit_shomi_dataset.py             # Taxonomic audit of external dataset components
    ├── stratify_by_source.py              # LVR stratification (Synthetic vs. External)
    └── separability_baseline.py           # Linear separability baseline (TF-IDF + Logistic Regression)
