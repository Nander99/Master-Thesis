import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def arc_statistics(df, dataset_name="ARC", output_dir="arc_stats"):
    os.makedirs(output_dir, exist_ok=True)
    stats = {
        "Total questions": len(df),
        "Average question length (words)": df["question"].apply(lambda x: len(str(x).split())).mean(),
        "Average number of words per choice": df["choices"].apply(
            lambda x: sum(len(str(c).split()) for c in eval(x)) / len(eval(x)) if isinstance(x, str) else 0
        ).mean(),
        "Class distribution": df["grade_group"].value_counts(normalize=True).to_dict()
    }

    print(f"\n=== {dataset_name} Dataset Statistics ===")
    for k, v in stats.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")

    path = os.path.join(output_dir, f"{dataset_name.lower().replace(' ', '_')}_stats.txt")
    with open(path, "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v:.2f}\n" if isinstance(v, float) else f"{k}: {v}\n")
    print(f"[Saved] Text stats → {path}")

if __name__ == "__main__":
    train_df = pd.read_csv("arc_processed/arc_train_preprocessed.csv")
    test_df = pd.read_csv("arc_processed/arc_test_preprocessed.csv")
    arc_statistics(train_df, dataset_name="ARC Train")
    arc_statistics(test_df, dataset_name="ARC Test")


