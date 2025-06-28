import pandas as pd
import torch
import math
from torch.utils.data import DataLoader
from transformers import AutoModelForMultipleChoice, AutoTokenizer
import torch.nn.functional as F
from tqdm import tqdm

# Load model and tokenizer
model_dir = "race_model"
model = AutoModelForMultipleChoice.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model.eval().cuda()

# Load test data
test_df = pd.read_csv("race_data/race_test.csv")

# Dataset class (matches training format with summary + question + answer)
class RaceMCQDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = []
        for _, row in dataframe.iterrows():
            summary = str(row["summary"]).strip()
            question = str(row["question"]).strip()
            choices = [str(x).strip() for x in eval(row["choices"])]
            answer_key = row["answer"].strip().upper()
            key_map = {"A": 0, "B": 1, "C": 2, "D": 3}

            if answer_key not in key_map or len(choices) != 4:
                continue

            label = key_map[answer_key]
            inputs = [f"{summary} Question: {question} Answer: {choice}" for choice in choices]

            encoding = tokenizer(
                inputs,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            self.data.append((encoding, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Prepare dataset and loader
dataset = RaceMCQDataset(test_df, tokenizer)
loader = DataLoader(dataset, batch_size=1)

# Compute entropy and confidence
all_entropy, all_confidence, all_true_labels = [], [], []

with torch.no_grad():
    for batch in tqdm(loader):
        encoding, label = batch
        input_ids = encoding["input_ids"].cuda()
        attention_mask = encoding["attention_mask"].cuda()
        label = label.item()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze(0)
        probs = F.softmax(logits, dim=-1)

        entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
        max_entropy = math.log(len(probs))
        normalized_entropy = entropy / max_entropy

        confidence = probs[label].item()

        all_entropy.append(normalized_entropy)
        all_confidence.append(confidence)
        all_true_labels.append(label)

# Save new dataset
results_df = test_df.iloc[:len(all_entropy)].copy()
results_df["entropy"] = all_entropy
results_df["confidence_correct"] = all_confidence
results_df["true_label"] = all_true_labels
results_df.to_csv("race_data/race_confidence.csv", index=False)
print("Saved RACE dataset with entropy and confidence.")
