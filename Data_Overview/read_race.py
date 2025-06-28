import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_processed_race():
    train_df = pd.read_csv("race_processed/race_train_sampled_summarized.csv")
    test_df = pd.read_csv("race_processed/race_test_sampled_summarized.csv")
    return train_df, test_df


def race_statistics(df, dataset_name="RACE", output_dir="race_stats"):
    os.makedirs(output_dir, exist_ok=True)
    stats = {
        "Total questions": len(df),
        "Avg. question length (words)": df["question"].apply(lambda x: len(str(x).split())).mean(),
        "Avg. passage length (words)": df["passage"].apply(lambda x: len(str(x).split())).mean(),
        "Avg. words per choice": df["choices"].apply(
            lambda x: sum(len(str(c).split()) for c in eval(x)) / len(eval(x)) if isinstance(x, str) else 0
        ).mean(),
        "Class distribution": df["difficulty"].value_counts(normalize=True).to_dict()
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
    train_df, test_df = load_processed_race()
    race_statistics(train_df, dataset_name="RACE Train")
    race_statistics(test_df, dataset_name="RACE Test")

