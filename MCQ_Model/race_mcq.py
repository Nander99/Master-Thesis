import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from sklearn.metrics import classification_report, accuracy_score
set_seed(42)


def preprocess_dataframe(df, tokenizer, max_length=512):
    encodings, labels = [], []
    key_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    for _, row in df.iterrows():
        summary = str(row["summary"]).strip()
        question = str(row["question"]).strip()
        choices = [str(x).strip() for x in eval(row["choices"])]
        answer_key = row["answer"].strip().upper()

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
        encodings.append(encoding)
        labels.append(label)

    return encodings, labels

class RaceMCQDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.encodings[idx]
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

def evaluate(trainer, dataset, output_file, split_name="Test"):
    predictions = trainer.predict(dataset)
    y_pred = predictions.predictions.argmax(-1)
    y_true = predictions.label_ids

    report = classification_report(y_true, y_pred, digits=4)
    accuracy = accuracy_score(y_true, y_pred)

    with open(output_file, "a") as f:
        f.write(f"\n=== Classification Report ({split_name}) ===\n")
        f.write(report)
        f.write(f"\nAccuracy: {accuracy:.4f}\n")

    print(f"{split_name} evaluation complete. Accuracy: {accuracy:.4f}")

def train_race_model(train_df, test_df, model_name="bert-base-uncased", save_dir="race_model"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMultipleChoice.from_pretrained(model_name)

    train_encodings, train_labels = preprocess_dataframe(train_df, tokenizer)
    test_encodings, test_labels = preprocess_dataframe(test_df, tokenizer)

    train_dataset = RaceMCQDataset(train_encodings, train_labels)
    test_dataset = RaceMCQDataset(test_encodings, test_labels)

    training_args = TrainingArguments(
        output_dir=save_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir=os.path.join(save_dir, "logs"),
        logging_strategy="no",
        report_to="none",
        fp16=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        group_by_length=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Best model saved to: {save_dir}")

    results_file = os.path.join(save_dir, "final_results.txt")
    evaluate(trainer, train_dataset, results_file, split_name="Train")
    evaluate(trainer, test_dataset, results_file, split_name="Test")

if __name__ == "__main__":
    train_df = pd.read_csv("race_data/race_train.csv")
    test_df = pd.read_csv("race_data/race_test.csv")
    train_race_model(train_df, test_df)
