import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import os
import torch
nltk.download('punkt')
device = "cuda" if torch.cuda.is_available() else "cpu"
sent_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

def load_race_full():
    race_middle = load_dataset('ehovy/race', 'middle')
    race_high = load_dataset('ehovy/race', 'high')

    middle_df = pd.concat([pd.DataFrame(race_middle[split]) for split in ['train', 'validation', 'test']])
    high_df = pd.concat([pd.DataFrame(race_high[split]) for split in ['train', 'validation', 'test']])

    middle_df["difficulty"] = "middle"
    high_df["difficulty"] = "high"

    df = pd.concat([middle_df, high_df], ignore_index=True)
    df = df.rename(columns={"example_id": "id", "article": "passage", "options": "choices"})
    df = df.dropna(subset=["question", "choices", "answer", "passage"])
    df = df[df["choices"].apply(lambda x: isinstance(x, list) and len(x) == 4)]

    return df

def sample_one_question_per_article(df):
    return df.groupby("passage", group_keys=False).apply(lambda x: x.sample(1, random_state=42)).reset_index(drop=True)

def summarize_race(passage, question, top_k=3):
    sentences = sent_tokenize(passage)
    if len(sentences) <= top_k:
        return " ".join(sentences)
    question_emb = sent_model.encode(question, convert_to_tensor=True)
    sent_embs = sent_model.encode(sentences, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(question_emb, sent_embs).squeeze()
    if scores.dim() == 0:
        return sentences[0]
    top_indices = scores.argsort(descending=True)[:top_k]
    return " ".join([sentences[i] for i in top_indices])

def apply_summary(df, batch_size=64):
    summaries = []
    for i in tqdm(range(0, len(df), batch_size), desc="Summarizing"):
        batch = df.iloc[i:i+batch_size]
        for _, row in batch.iterrows():
            summaries.append(summarize_race(row["passage"], row["question"]))
    df["summary"] = summaries
    return df

if __name__ == "__main__":
    df = load_race_full()
    df = sample_one_question_per_article(df)

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["difficulty"], random_state=42)

    train_df = apply_summary(train_df)
    test_df = apply_summary(test_df)

    os.makedirs("race_processed", exist_ok=True)
    train_df.to_csv("race_processed/race_train_sampled_summarized.csv", index=False)
    test_df.to_csv("race_processed/race_test_sampled_summarized.csv", index=False)
    print("[Saved] Preprocessed RACE datasets to 'race_processed/'")