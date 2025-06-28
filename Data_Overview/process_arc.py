import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

def load_json_data(base_path):
    rows = []
    for difficulty in ["Easy", "Challenge"]:
        for split in ["Train", "Dev", "Test"]:
            jsonl_file = os.path.join(base_path, f"ARC-{difficulty}-{split}.jsonl")
            if not os.path.exists(jsonl_file):
                continue
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    q = data.get("question", {})
                    row = {
                        "id": data.get("id"),
                        "question": q.get("stem"),
                        "choices": [c["text"] for c in q.get("choices", [])],
                        "answer_key": data.get("answerKey"),
                        "difficulty": difficulty.lower()
                    }
                    rows.append(row)
    return pd.DataFrame(rows)

def load_csv_grades(base_path):
    grade_rows = []
    for difficulty in ["Easy", "Challenge"]:
        for split in ["Train", "Dev", "Test"]:
            csv_path = os.path.join(base_path, f"ARC-{difficulty}-{split}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df = df.rename(columns={"questionID": "id", "schoolGrade": "grade"})
                grade_rows.append(df[["id", "grade"]])
    return pd.concat(grade_rows, ignore_index=True)

def assign_grade_group(grade):
    try:
        g = int(grade)
        if 3 <= g <= 5:
            return "lower"
        elif 7 <= g <= 9:
            return "upper"
        else:
            return None
    except:
        return None

def get_answer_text(row):
    labels = ['A', 'B', 'C', 'D']
    answer = row['answer_key']
    try:
        if isinstance(answer, str) and answer.strip().upper() in labels:
            index = labels.index(answer.strip().upper())
        elif isinstance(answer, str) and answer.strip().isdigit():
            index = int(answer.strip()) - 1
        else:
            return None
        if 0 <= index < len(row['choices']):
            return row['choices'][index]
    except:
        pass
    return None

if __name__ == "__main__":
    base_path = "arc_data"

    json_df = load_json_data(base_path)
    csv_df = load_csv_grades(base_path)
    df = pd.merge(json_df, csv_df, on="id", how="inner")
    df = df.dropna(subset=["question", "choices", "answer_key", "grade"])

    df["grade_group"] = df["grade"].apply(assign_grade_group)
    df = df.dropna(subset=["grade_group"])

    df["answer_text"] = df.apply(get_answer_text, axis=1)
    df = df.dropna(subset=["answer_text"])
    df = df[df['choices'].apply(len) == 4].reset_index(drop=True)

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["grade_group"], random_state=42)

    os.makedirs("arc_processed", exist_ok=True)
    train_df.to_csv("arc_processed/arc_train_preprocessed.csv", index=False)
    test_df.to_csv("arc_processed/arc_test_preprocessed.csv", index=False)
    print("[Saved] Preprocessed train/test CSVs to arc_processed/")
