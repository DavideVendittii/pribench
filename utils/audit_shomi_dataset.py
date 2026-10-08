"""
Dataset Composition & Shomi Audit Tool (PriBench)

Usage:
    python utils/audit_shomi_dataset.py
"""

import os
import re
import pandas as pd

try:
    from datasets import load_dataset
except ImportError:
    os.system("pip install datasets pandas")
    from datasets import load_dataset


def main():
    ds = load_dataset("Shomi28/prompt-injection-dataset")

    split_name = "train" if "train" in ds else list(ds.keys())[0]
    df = ds[split_name].to_pandas()
    df.columns = df.columns.str.strip().str.lower()

    if "label_name" in df.columns:
        df_inj = df[df["label_name"].astype(str).str.lower() == "injection"].copy()
    elif "label" in df.columns:
        df_inj = df[df["label"] == 1].copy()
    else:
        df_inj = df.copy()

    privacy_patterns = [
        r"\bpassword(s)?\b", r"\bcredential(s)?\b", r"\bapi[-_]?key(s)?\b",
        r"\bsecret(s)?\b", r"\btoken(s)?\b", r"\bssn\b", r"\bsocial security\b",
        r"\bcredit card\b", r"\bbank account\b", r"\bprivate key\b", r"\bpii\b",
        r"\bpersonal data\b", r"\buser info(rmation)?\b", r"\bdatabase\b",
        r"\bemail address(es)?\b", r"\bphone number(s)?\b", r"\bconfidential\b",
        r"\bauth(entication)?\b", r"\blogin\b", r"\bpassport\b", r"\bsystem prompt\b",
        r"\binitial prompt\b", r"\bsystem instruction(s)?\b", r"\bdeveloper prompt\b",
        r"\bhidden instructions\b", r"\bleak\b", r"\bdump\b"
    ]

    regex_privacy = re.compile("|".join(privacy_patterns), re.IGNORECASE)

    def classify_prompt(text):
        if regex_privacy.search(str(text)):
            return "Private Data / Secret Extraction"
        return "Generic Injection / System Override"

    df_inj["risk_category"] = df_inj["text"].apply(classify_prompt)

    counts = df_inj["risk_category"].value_counts()
    total = len(df_inj)
    pcts = (counts / total * 100).round(2)

    n_priv = counts.get("Private Data / Secret Extraction", 0)
    pct_priv = pcts.get("Private Data / Secret Extraction", 0.0)
    n_inj = counts.get("Generic Injection / System Override", 0)
    pct_inj = pcts.get("Generic Injection / System Override", 0.0)

    output_dir = "shomi_audit_results"
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "shomi_classified.csv")
    df_inj.to_csv(csv_path, index=False)

    latex_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"\textbf{Source} & \textbf{Risk Category} & \textbf{Count ($N$)} & \textbf{\%} \\",
        r"\midrule",
        r"Synthetic & Direct Memory Requests & 834 & 26.5\% \\",
        r"Synthetic & Indirect Attribute Inference & 834 & 26.5\% \\",
        r"Synthetic & Social Engineering / Delegation & 834 & 26.5\% \\",
        r"\midrule",
        r"Shomi (External) & Private Data / Secret Extraction & " + str(n_priv) + r" & " + str(round(pct_priv, 1)) + r"\% \\",
        r"Shomi (External) & Generic Injection / System Override & " + str(n_inj) + r" & " + str(round(pct_inj, 1)) + r"\% \\",
        r"\midrule",
        r"\textbf{Total Positive Class} & & \textbf{3,142} & \textbf{100.0\%} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Taxonomic composition of the positive class ($label=1$) stratified by generation source.}",
        r"\label{tab:dataset_composition}",
        r"\end{table}"
    ]

    tex_path = os.path.join(output_dir, "dataset_composition_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))


if __name__ == "__main__":
    main()