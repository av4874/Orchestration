"""
Kaggle retrain script — injection detector v4
Run on Kaggle GPU (T4 or P100): Settings > Accelerator > GPU

Dataset input: adversarial-guardrail-injection-r1 (attach in Kaggle notebook)
Output: fine-tuned DistilBERT model pushed to HF Hub as Builder117/distilbert-prompt-injection-v4
"""
import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    DistilBertTokenizerFast, DistilBertForSequenceClassification,
    TrainingArguments, Trainer,
)
import torch

HF_TOKEN = os.environ["HF_TOKEN"]   # set in Kaggle Secrets
MODEL_OUT = "Builder117/distilbert-prompt-injection-v4"
DATASET_CSV = "/kaggle/input/adversarial-guardrail-injection-r1/dataset.csv"
BASE_MODEL = "distilbert-base-uncased"
LABEL2ID = {"INJECTION": 1, "LEGIT": 0}
ID2LABEL = {1: "INJECTION", 0: "LEGIT"}

# Load dataset
df = pd.read_csv(DATASET_CSV)
df = df[["text", "label"]].dropna()
df["label"] = df["label"].astype(int)

# Train/val split
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=42)

# Tokenize
tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)

ds = DatasetDict({
    "train": Dataset.from_pandas(train_df).map(tokenize, batched=True),
    "validation": Dataset.from_pandas(val_df).map(tokenize, batched=True),
})

# Model
model = DistilBertForSequenceClassification.from_pretrained(
    BASE_MODEL, num_labels=2, label2id=LABEL2ID, id2label=ID2LABEL
)

# Training
args = TrainingArguments(
    output_dir=f"/kaggle/working/{MODEL_OUT}",
    num_train_epochs=6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    push_to_hub=True,
    hub_model_id=MODEL_OUT,
    hub_token=HF_TOKEN,
)

trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"])
trainer.train()
trainer.push_to_hub()
print(f"Model pushed: {MODEL_OUT}")
