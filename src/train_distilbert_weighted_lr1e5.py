import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)


# set project paths and create unique output folders
PROJECT_ROOT = (
    r"C:\Main_Drive\work\workarea\projects"
    r"\research_manuscript_quality_analyzer"
)

train_path = PROJECT_ROOT + r"\data\raw\cola_train.csv"
val_path = PROJECT_ROOT + r"\data\raw\cola_validation.csv"

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

model_path = (
    PROJECT_ROOT + rf"\models\distilbert_weighted_lr1e5\run_{run_id}"
)
report_dir = (
    PROJECT_ROOT + rf"\reports\distilbert_weighted_lr1e5\run_{run_id}"
)

os.makedirs(model_path, exist_ok=False)
os.makedirs(report_dir, exist_ok=False)

print(f"Experiment ID: {run_id}")
print(f"Model folder: {model_path}")
print(f"Report folder: {report_dir}")


# load and clean the dataset
def load_data(file_path):
    df = pd.read_csv(file_path, usecols=["sentence", "label"])
    df = df.dropna(subset=["sentence", "label"]).copy()
    df["sentence"] = df["sentence"].str.strip()
    df = df[df["sentence"] != ""].copy()
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)

    # remove conflicting labels and duplicate sentences
    label_counts = df.groupby("sentence")["label"].nunique()
    conflicting = label_counts[label_counts > 1].index
    df = df[~df["sentence"].isin(conflicting)].copy()
    df = df.drop_duplicates(subset=["sentence"], keep="first")

    return df.reset_index(drop=True)


train_df = load_data(train_path)
val_df = load_data(val_path)

print(f"Training examples: {len(train_df)}")
print(f"Validation examples: {len(val_df)}")


# calculate balanced class weights from training data only
classes = np.array([0, 1])

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["label"].to_numpy()
)

class_weights = torch.tensor(class_weights, dtype=torch.float32)

print(f"Class weights: {class_weights.tolist()}")


# load the pretrained model and tokenizer
model_name = "distilbert/distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)


# tokenize sentences
def tokenize_data(batch):
    return tokenizer(
        batch["sentence"],
        truncation=True,
        max_length=128
    )


train_dataset = Dataset.from_pandas(train_df).map(
    tokenize_data,
    batched=True,
    remove_columns=["sentence"]
)

val_dataset = Dataset.from_pandas(val_df).map(
    tokenize_data,
    batched=True,
    remove_columns=["sentence"]
)

train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# calculate validation metrics
def calculate_metrics(result):
    predictions = np.argmax(result.predictions, axis=1)
    labels = result.label_ids

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1
    }


# apply weighted cross-entropy during training
class WeightedTrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        weights = class_weights.to(outputs.logits.device)

        loss_function = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_function(outputs.logits, labels)

        if return_outputs:
            return loss, outputs

        return loss


# configure training with a lower learning rate and early stopping
training_args = TrainingArguments(
    output_dir=report_dir + r"\training_output",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=1e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_macro_f1",
    greater_is_better=True,
    save_total_limit=1,
    report_to="none",
    use_cpu=True,
    seed=42
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=calculate_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=1,
            early_stopping_threshold=0.001
        )
    ]
)


# train and measure execution time
start_time = time.time()
trainer.train()
training_time = time.time() - start_time


# evaluate the selected best checkpoint
evaluation_results = trainer.evaluate()
predictions = trainer.predict(val_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)

matrix = confusion_matrix(
    val_df["label"],
    predicted_labels,
    labels=[0, 1]
)

class_report = classification_report(
    val_df["label"],
    predicted_labels,
    labels=[0, 1],
    target_names=["unacceptable", "acceptable"],
    zero_division=0
)

print("\nWeighted DistilBERT with learning rate 1e-5:")

for metric, value in evaluation_results.items():
    if metric.startswith("eval_"):
        print(f"{metric}: {value}")

print(f"\nTraining time (seconds): {training_time:.2f}")
print("\nConfusion matrix:")
print(matrix)
print("\nClassification report:")
print(class_report)


# save the best model and tokenizer
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)


# save a new report without overwriting previous reports
report_path = report_dir + r"\distilbert_weighted_lr1e5_report.md"

with open(report_path, "x", encoding="utf-8") as file:
    file.write("# Class-Weighted DistilBERT with Learning Rate 1e-5\n\n")
    file.write(f"- Experiment ID: `{run_id}`\n")
    file.write("- Model: DistilBERT base uncased\n")
    file.write("- Loss: balanced class-weighted cross-entropy\n")
    file.write("- Learning rate: `1e-5`\n")
    file.write("- Maximum epochs: 4\n")
    file.write("- Training batch size: 8\n")
    file.write("- Maximum sequence length: 128 tokens\n")
    file.write("- Early stopping: patience 1 based on validation macro F1\n")
    file.write(f"- Training examples: {len(train_df)}\n")
    file.write(f"- Validation examples: {len(val_df)}\n")
    file.write(f"- Training time: {training_time:.2f} seconds\n\n")

    file.write("## Class Weights\n\n")
    file.write(f"- Unacceptable (0): {class_weights[0].item():.4f}\n")
    file.write(f"- Acceptable (1): {class_weights[1].item():.4f}\n")

    file.write("\n## Validation Metrics\n\n")
    for metric, value in evaluation_results.items():
        if metric.startswith("eval_"):
            file.write(f"- **{metric}:** {value:.4f}\n")

    file.write("\n## Confusion Matrix\n\n")
    file.write("Rows = actual labels `[0, 1]`; columns = predicted labels `[0, 1]`.\n\n")
    file.write(f"```text\n{matrix}\n```\n")

    file.write("\n## Classification Report\n\n")
    file.write(f"```text\n{class_report}\n```\n")

    file.write(
        "\n## Limitations\n\n"
        "This model evaluates linguistic acceptability on CoLA. It does not "
        "directly assess factual correctness, methodology, citation quality, "
        "originality, or overall manuscript quality.\n"
    )

print(f"\nBest model saved: {model_path}")
print(f"New report saved: {report_path}")
