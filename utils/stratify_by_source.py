"""
Source Stratification Tool (PriBench)

Usage:
    python utils/stratify_by_source.py
"""

import os
import glob
import pandas as pd


def load_shomi_prompts(shomi_csv_path):
    if os.path.exists(shomi_csv_path):
        df_shomi = pd.read_csv(shomi_csv_path)
        return set(df_shomi["text"].astype(str).str.strip())
    return set()


def main():
    shomi_csv = "shomi_audit_results/shomi_classified.csv"
    shomi_prompts = load_shomi_prompts(shomi_csv)

    result_files = glob.glob("./**/benchmark_results_raw_vs_star.csv", recursive=True)

    if not result_files:
        print("No benchmark_results_raw_vs_star.csv files found.")
        return

    results_by_model = {}

    for filepath in sorted(result_files):
        folder_name = os.path.basename(os.path.dirname(filepath))
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip().str.lower()

        pos_df = df[df["true_label"] == 1].copy()

        pos_df["source"] = pos_df["prompt"].astype(str).str.strip().apply(
            lambda p: "Shomi" if p in shomi_prompts else "Synthetic"
        )

        metrics = {}
        for src in ["Synthetic", "Shomi"]:
            sub = pos_df[pos_df["source"] == src]
            if len(sub) > 0:
                raw_lvr = (1.0 - sub["raw_refused"].astype(bool).mean()) * 100.0
                star_lvr = (1.0 - sub["star_refused"].astype(bool).mean()) * 100.0
                metrics[src] = {
                    "count": len(sub),
                    "raw_lvr": round(raw_lvr, 1),
                    "star_lvr": round(star_lvr, 1),
                }
            else:
                metrics[src] = {"count": 0, "raw_lvr": 0.0, "star_lvr": 0.0}

        results_by_model[folder_name] = metrics

    latex_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Condition / Setting} & \textbf{Synthetic ($N=2,502$)} & \textbf{Shomi ($N=517$)} \\",
        r"\midrule",
    ]

    for model_name, data in results_by_model.items():
        raw_syn = data["Synthetic"]["raw_lvr"]
        raw_sh = data["Shomi"]["raw_lvr"]
        star_syn = data["Synthetic"]["star_lvr"]
        star_sh = data["Shomi"]["star_lvr"]

        latex_lines.extend([
            r"\multirow{2}{*}{\textbf{" + model_name + r"}}",
            r" & Baseline (Direct Interaction) & " + str(raw_syn) + r"\% & " + str(raw_sh) + r"\% \\",
            r" & Guarded (STAR Intervention) & " + str(star_syn) + r"\% & " + str(star_sh) + r"\% \\",
            r"\midrule",
        ])

    if latex_lines[-1] == r"\midrule":
        latex_lines[-1] = r"\bottomrule"

    latex_lines.extend([
        r"\end{tabular}",
        r"\caption{Leakage Vulnerability Rate (LVR) stratified by prompt source across evaluation conditions.}",
        r"\label{tab:source_stratification}",
        r"\end{table}",
    ])

    os.makedirs("shomi_audit_results", exist_ok=True)
    out_tex_path = "shomi_audit_results/source_stratification_table.tex"
    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))


if __name__ == "__main__":
    main()