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

class MCQDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=256):
        self.encodings = []
        self.labels = []
        for _, row in dataframe.iterrows():
            q = row["question"]
            c = [str(x).strip() for x in eval(row["choices"])]
            a = str(row["answer_text"]).strip()

            try:
                label = c.index(a)
            except ValueError:
                if "answer_key" in row:
                    key = row["answer_key"].strip().upper()
                    key_map = {"A": 0, "B": 1, "C": 2, "D": 3}
                    if key in key_map:
                        label = key_map[key]
                    else:
                        continue
                else:
                    continue

            inputs = [f"Question: {q} Answer: {choice}" for choice in c]
            encoding = tokenizer(
                inputs,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            self.encodings.append(encoding)
            self.labels.append(label)

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

def train_model(train_df, test_df, model_name="bert-base-uncased", save_dir="arc_model"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMultipleChoice.from_pretrained(model_name)

    train_dataset = MCQDataset(train_df, tokenizer)
    test_dataset = MCQDataset(test_df, tokenizer)

    training_args = TrainingArguments(
        output_dir=save_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        per_device_train_batch_size=16,       
        per_device_eval_batch_size=16,       
        num_train_epochs=10,                 
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir=os.path.join(save_dir, "logs"),
        logging_strategy="epoch",
        report_to="none",
        fp16=True,                            
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
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
    train_df = pd.read_csv("arc_data/arc_train.csv")
    test_df = pd.read_csv("arc_data/arc_test.csv")
    train_model(train_df, test_df, "microsoft/deberta-v3-base")
