
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.model_selection import train_test_split
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


# set project paths and unique output folders
PROJECT_ROOT = (
    r"C:\Main_Drive\work\workarea\projects"
    r"\research_manuscript_quality_analyzer"
)

data_dir = PROJECT_ROOT + r"\data\processed\final_evaluation"
development_path = data_dir + r"\development.csv"
holdout_path = data_dir + r"\final_holdout.csv"

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

model_dir = (
    PROJECT_ROOT
    + rf"\models\distilbert_final\run_{run_id}"
)
report_dir = (
    PROJECT_ROOT
    + rf"\reports\distilbert_final\run_{run_id}"
)

os.makedirs(model_dir, exist_ok=False)
os.makedirs(report_dir, exist_ok=False)

print(f"Experiment ID: {run_id}")


# load and validate the saved datasets
development_df = pd.read_csv(development_path)
holdout_df = pd.read_csv(holdout_path)

for df in [development_df, holdout_df]:
    df.dropna(subset=["sentence", "label"], inplace=True)
    df["sentence"] = df["sentence"].astype(str).str.strip()
    df.drop(df[df["sentence"] == ""].index, inplace=True)
    df["label"] = df["label"].astype(int)

if not set(development_df["label"].unique()).issubset({0, 1}):
    raise ValueError("Unexpected labels in development data.")

if not set(holdout_df["label"].unique()).issubset({0, 1}):
    raise ValueError("Unexpected labels in final holdout.")

print(f"Development examples: {len(development_df)}")
print(f"Final holdout examples: {len(holdout_df)}")


# create an internal validation set from development data only
train_df, val_df = train_test_split(
    development_df,
    test_size=0.15,
    random_state=42,
    stratify=development_df["label"]
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print(f"Training examples: {len(train_df)}")
print(f"Internal validation examples: {len(val_df)}")


# calculate class weights using training labels only
classes = np.array([0, 1])

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["label"].to_numpy()
)

class_weights = torch.tensor(weights, dtype=torch.float32)
print(f"Class weights: {class_weights.tolist()}")


# load the pretrained model and tokenizer
model_name = "distilbert/distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)


# tokenize the datasets
def tokenize_data(batch):
    return tokenizer(
        batch["sentence"],
        truncation=True,
        max_length=128
    )


def prepare_dataset(df):
    dataset = Dataset.from_pandas(df[["sentence", "label"]])
    dataset = dataset.map(
        tokenize_data,
        batched=True,
        remove_columns=["sentence"]
    )
    return dataset.rename_column("label", "labels")


train_dataset = prepare_dataset(train_df)
val_dataset = prepare_dataset(val_df)
holdout_dataset = prepare_dataset(holdout_df)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# calculate macro-averaged validation metrics
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


# apply class-weighted cross-entropy
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

        loss_function = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(outputs.logits.device)
        )

        loss = loss_function(outputs.logits, labels)

        if return_outputs:
            return loss, outputs

        return loss


# configure training and internal validation
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


# train using development data only
start_time = time.time()
trainer.train()
training_time = time.time() - start_time

print(f"Training time: {training_time:.2f} seconds")


# evaluate the selected model on the final holdout once
holdout_results = trainer.predict(holdout_dataset)
holdout_predictions = np.argmax(
    holdout_results.predictions,
    axis=1
)

holdout_labels = holdout_df["label"].to_numpy()

matrix = confusion_matrix(
    holdout_labels,
    holdout_predictions,
    labels=[0, 1]
)

class_report = classification_report(
    holdout_labels,
    holdout_predictions,
    labels=[0, 1],
    target_names=["unacceptable", "acceptable"],
    zero_division=0
)

precision, recall, f1, _ = precision_recall_fscore_support(
    holdout_labels,
    holdout_predictions,
    average="macro",
    zero_division=0
)

accuracy = accuracy_score(holdout_labels, holdout_predictions)

print("\nFinal holdout results")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro precision: {precision:.4f}")
print(f"Macro recall: {recall:.4f}")
print(f"Macro F1: {f1:.4f}")
print("\nConfusion matrix:")
print(matrix)
print("\nClassification report:")
print(class_report)


# save model and tokenizer without overwriting earlier experiments
trainer.save_model(model_dir)
tokenizer.save_pretrained(model_dir)


# save holdout metrics in a separate report
report_path = report_dir + r"\final_holdout_report.md"

with open(report_path, "x", encoding="utf-8") as file:
    file.write("# DistilBERT Final Holdout Evaluation\n\n")
    file.write(f"- Experiment ID: `{run_id}`\n")
    file.write("- Model: DistilBERT base uncased\n")
    file.write("- Training: class-weighted cross-entropy\n")
    file.write("- Learning rate: `1e-5`\n")
    file.write("- Maximum epochs: 4\n")
    file.write("- Training batch size: 8\n")
    file.write("- Maximum sequence length: 128 tokens\n")
    file.write(f"- Training examples: {len(train_df)}\n")
    file.write(f"- Internal validation examples: {len(val_df)}\n")
    file.write(f"- Final holdout examples: {len(holdout_df)}\n")
    file.write(f"- Training time: {training_time:.2f} seconds\n\n")

    file.write("## Final Holdout Metrics\n\n")
    file.write(f"- Accuracy: {accuracy:.4f}\n")
    file.write(f"- Macro precision: {precision:.4f}\n")
    file.write(f"- Macro recall: {recall:.4f}\n")
    file.write(f"- Macro F1: {f1:.4f}\n\n")

    file.write("## Confusion Matrix\n\n")
    file.write("Rows = actual labels `[0, 1]`; columns = predicted labels `[0, 1]`.\n\n")
    file.write(f"```text\n{matrix}\n```\n\n")

    file.write("## Classification Report\n\n")
    file.write(f"```text\n{class_report}\n```\n\n")

    file.write(
        "## Evaluation Notes\n\n"
        "The final holdout was separated before this experiment. "
        "The model was selected using an internal validation subset of "
        "the development data. This is a held-out evaluation of a new "
        "stratified split from CoLA, not the official CoLA test set. "
        "Results measure linguistic acceptability, not complete academic "
        "manuscript quality, factual correctness, citations, or originality.\n"
    )

print(f"\nModel saved: {model_dir}")
print(f"Report saved: {report_path}")
