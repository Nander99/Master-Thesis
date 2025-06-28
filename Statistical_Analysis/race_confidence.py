import pandas as pd
import ast
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr, spearmanr, ttest_ind

def get_answer_text_race(row):
    labels = ['A', 'B', 'C', 'D']
    answer_key = row["answer_key"]
    try:
        if isinstance(answer_key, str) and answer_key.strip().upper() in labels:
            index = labels.index(answer_key.strip().upper())
            return row["choices"][index] if 0 <= index < len(row["choices"]) else None
    except:
        return None
    return None

def convert_race_to_new_format(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    if isinstance(df.loc[0, "choices"], str) and df.loc[0, "choices"].startswith("["):
        df["choices"] = df["choices"].apply(ast.literal_eval)

    df["answer_text"] = df.apply(get_answer_text_race, axis=1)
    df = df.rename(columns={"answer": "answer_key"})

    keep_cols = ["id", "passage", "question", "choices", "answer_key", "difficulty", 
                 "answer_text", "entropy", "confidence_correct"]
    df = df.dropna(subset=keep_cols)
    df = df[keep_cols]
    return df

def interpret_corr(stat, p):
    direction = "positive" if stat > 0 else "negative"
    strength = "weak"
    abs_stat = abs(stat)
    if abs_stat > 0.5:
        strength = "strong"
    elif abs_stat > 0.3:
        strength = "moderate"
    elif abs_stat > 0.1:
        strength = "small"
    signif = "significant" if p < 0.05 else "not significant"
    return f"{strength} {direction} correlation ({signif}, r = {stat:.3f}, p = {p:.3g})"

def interpret_ttest(stat, p, var1_name, var2_name):
    if p < 0.05:
        direction = f"{var1_name} < {var2_name}" if stat < 0 else f"{var1_name} > {var2_name}"
        return f"significant difference ({direction}, t = {stat:.3f}, p = {p:.3g})"
    else:
        return f"no significant difference (t = {stat:.3f}, p = {p:.3g})"

def analyze_race_data(df, output_dir="race_confidence"):
    summary_stats = df.groupby("difficulty")[["entropy", "confidence_correct"]].mean()

    df["difficulty_numeric"] = df["difficulty"].map({"middle": 0, "high": 1})

    pearson_entropy = pearsonr(df["difficulty_numeric"], df["entropy"])
    pearson_conf = pearsonr(df["difficulty_numeric"], df["confidence_correct"])
    spearman_entropy = spearmanr(df["difficulty_numeric"], df["entropy"])
    spearman_conf = spearmanr(df["difficulty_numeric"], df["confidence_correct"])
    pearson_entropy_conf = pearsonr(df["entropy"], df["confidence_correct"])
    spearman_entropy_conf = spearmanr(df["entropy"], df["confidence_correct"])

    entropy_lower = df[df["difficulty"] == "middle"]["entropy"]
    entropy_upper = df[df["difficulty"] == "high"]["entropy"]
    t_entropy = ttest_ind(entropy_lower, entropy_upper, equal_var=False)

    conf_lower = df[df["difficulty"] == "middle"]["confidence_correct"]
    conf_upper = df[df["difficulty"] == "high"]["confidence_correct"]
    t_conf = ttest_ind(conf_lower, conf_upper, equal_var=False)

    with open(f"{output_dir}/race_confidence_analysis.txt", "w") as f:
        f.write("=== Summary of Entropy and Confidence by Difficulty ===\n")
        f.write(summary_stats.to_string())
        f.write("\n\n=== Correlation Analysis ===\n")
        f.write("Entropy vs Difficulty (Pearson): " + interpret_corr(*pearson_entropy) + "\n")
        f.write("Confidence vs Difficulty (Pearson): " + interpret_corr(*pearson_conf) + "\n")
        f.write("Entropy vs Difficulty (Spearman): " + interpret_corr(*spearman_entropy) + "\n")
        f.write("Confidence vs Difficulty (Spearman): " + interpret_corr(*spearman_conf) + "\n")
        f.write("\n=== Correlation Between Entropy and Confidence ===\n")
        f.write("Pearson: " + interpret_corr(*pearson_entropy_conf) + "\n")
        f.write("Spearman: " + interpret_corr(*spearman_entropy_conf) + "\n")
        f.write("\n\n=== T-Test Results ===\n")
        f.write("T-test for Entropy (middle vs high): " + interpret_ttest(t_entropy.statistic, t_entropy.pvalue, "middle", "high") + "\n")
        f.write("T-test for Confidence (middle vs high): " + interpret_ttest(t_conf.statistic, t_conf.pvalue, "middle", "high") + "\n")

    sns.scatterplot(x="entropy", y="confidence_correct", data=df)
    plt.title("Scatterplot: Entropy vs. Confidence of Correct Answer (RACE)")
    plt.xlabel("Entropy")
    plt.ylabel("Confidence of Correct Answer")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/race_scatter_entropy_vs_confidence.png")
    plt.clf()

if __name__ == "__main__":
    os.makedirs("race_stats", exist_ok=True)
    input_csv = "race_confidence/race_confidence.csv"
    output_csv = "race_confidence/race_confidence.csv"
    output_dir = "race_stats"

    df = convert_race_to_new_format(input_csv, output_csv)
    analyze_race_data(df, output_dir=output_dir)
