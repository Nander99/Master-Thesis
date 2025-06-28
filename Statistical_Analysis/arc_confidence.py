import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr, spearmanr, ttest_ind

# Load the dataset
os.makedirs("arc_stats", exist_ok=True)
df = pd.read_csv("arc_confidence/arc_confidence.csv")

# Summary Statistics by Grade Group
summary_stats = df.groupby("grade_group")[["entropy", "confidence_correct"]].mean()

# Map difficulty group to numeric
difficulty_map = {"lower": 0, "upper": 1}
df["difficulty_numeric"] = df["grade_group"].map(difficulty_map)

# Correlations
pearson_entropy = pearsonr(df["difficulty_numeric"], df["entropy"])
pearson_conf = pearsonr(df["difficulty_numeric"], df["confidence_correct"])
spearman_entropy = spearmanr(df["difficulty_numeric"], df["entropy"])
spearman_conf = spearmanr(df["difficulty_numeric"], df["confidence_correct"])

# Correlation between Entropy and Confidence 
pearson_entropy_conf = pearsonr(df["entropy"], df["confidence_correct"])
spearman_entropy_conf = spearmanr(df["entropy"], df["confidence_correct"])

# T-tests 
lower = df[df["grade_group"] == "lower"]
upper = df[df["grade_group"] == "upper"]

t_entropy = ttest_ind(lower["entropy"], upper["entropy"], equal_var=False)
t_conf = ttest_ind(lower["confidence_correct"], upper["confidence_correct"], equal_var=False)

# Explaining correlation
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
    signif = "significant" if p < 0.001 else "not significant"
    return f"{strength} {direction} correlation ({signif}, r = {stat:.3f}, p = {p:.3g})"

# Explaining t-test
def interpret_ttest(t, p, variable, group1="lower", group2="upper"):
    signif = "significant" if p < 0.001 else "not significant"
    direction = f"{group1} < {group2}" if t < 0 else f"{group1} > {group2}"
    return f"{variable}: {signif} difference ({direction}, t = {t:.3f}, p = {p:.3g})"

# Analysis summary
with open("arc_stats/arc_confidence_analysis.txt", "w") as f:
    f.write("=== Summary of Entropy and Confidence by Grade Group ===\n")
    f.write(summary_stats.to_string())
    f.write("\n\n=== Correlation Analysis ===\n")
    f.write("Entropy vs Grade Group (Pearson): " + interpret_corr(*pearson_entropy) + "\n")
    f.write("Confidence vs Grade Group (Pearson): " + interpret_corr(*pearson_conf) + "\n")
    f.write("Entropy vs Grade Group (Spearman): " + interpret_corr(*spearman_entropy) + "\n")
    f.write("Confidence vs Grade Group (Spearman): " + interpret_corr(*spearman_conf) + "\n")
    f.write("\n=== Correlation Between Entropy and Confidence ===\n")
    f.write("Pearson: " + interpret_corr(*pearson_entropy_conf) + "\n")
    f.write("Spearman: " + interpret_corr(*spearman_entropy_conf) + "\n")
    f.write("\n=== T-Test Results ===\n")
    f.write(interpret_ttest(t_entropy.statistic, t_entropy.pvalue, "Entropy") + "\n")
    f.write(interpret_ttest(t_conf.statistic, t_conf.pvalue, "Confidence") + "\n")

# Scatterplot: Entropy vs Confidence
plt.figure(figsize=(6, 5))
sns.scatterplot(x="entropy", y="confidence_correct", data=df)
plt.title("Scatterplot: Entropy vs. Confidence of Correct Answer (ARC)")
plt.xlabel("Entropy")
plt.ylabel("Confidence of Correct Answer")
plt.tight_layout()
plt.savefig("arc_stats/arc_scatter_entropy_vs_confidence.png")
plt.clf()