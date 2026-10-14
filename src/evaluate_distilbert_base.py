import os
import time
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# set project paths
PROJECT_ROOT = r"C:\Main_Drive\work\workarea\projects\research_manuscript_quality_analyzer"

val_path = PROJECT_ROOT + r"\data\raw\cola_validation.csv"
model_path = PROJECT_ROOT + r"\models\distilbert_base"
report_dir = PROJECT_ROOT + r"\reports\distilbert_base"

os.makedirs(model_path, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)

# load validation data
df = pd.read_csv(val_path, usecols=["sentence", "label"])
df = df.dropna(subset=["sentence", "label"]).copy()
df["sentence"] = df["sentence"].str.strip()
df = df[df["sentence"] != ""].copy()
df = df[df["label"].isin([0, 1])].copy()
df["label"] = df["label"].astype(int)
df = df.reset_index(drop=True)

# load pretrained distilbert with a new classification head
model_name = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# tokenize validation sentences
def tokenize_data(batch):
    return tokenizer(batch["sentence"], truncation=True, max_length=128)

dataset = Dataset.from_pandas(df).map(
    tokenize_data,
    batched=True,
    remove_columns=["sentence"]
)
dataset = dataset.rename_column("label", "labels")

# calculate evaluation metrics
def calculate_metrics(result):
    predictions = np.argmax(result.predictions, axis=1)
    labels = result.label_ids

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1
    }

# configure evaluation without training
training_args = TrainingArguments(
    output_dir=report_dir + r"\evaluation_output",
    per_device_eval_batch_size=16,
    report_to="none",
    use_cpu=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=dataset,
    compute_metrics=calculate_metrics
)

# evaluate the unfine-tuned model
start_time = time.time()
results = trainer.evaluate()
evaluation_time = time.time() - start_time

predictions = trainer.predict(dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)

matrix = confusion_matrix(df["label"], predicted_labels, labels=[0, 1])
class_report = classification_report(
    df["label"],
    predicted_labels,
    labels=[0, 1],
    zero_division=0
)

print("\nPretrained DistilBERT baseline results:")
for metric, value in results.items():
    if metric.startswith("eval_"):
        print(f"{metric}: {value}")

print(f"evaluation_time_seconds: {evaluation_time:.2f}")
print("\nConfusion matrix:")
print(matrix)
print("\nClassification report:")
print(class_report)

# save the original pretrained model and tokenizer separately
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# save baseline report
report_path = report_dir + r"\distilbert_base_report.md"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("# Pretrained DistilBERT Baseline Report\n\n")
    file.write("## Experimental Setup\n\n")
    file.write("- Model: DistilBERT base uncased\n")
    file.write("- Fine-tuning: Not performed\n")
    file.write("- Classification head: Newly initialized and untrained\n")
    file.write("- Task: Binary linguistic acceptability classification\n")
    file.write(f"- Validation examples: {len(df)}\n")
    file.write(f"- Evaluation time (seconds): {evaluation_time:.2f}\n\n")

    file.write("## Validation Metrics\n\n")
    for metric, value in results.items():
        if metric.startswith("eval_"):
            file.write(f"- **{metric}:** {value:.4f}\n")

    file.write("\n## Confusion Matrix\n\n")
    file.write("Rows = actual labels `[0, 1]`; columns = predicted labels `[0, 1]`.\n\n")
    file.write(f"```text\n{matrix}\n```\n")

    file.write("\n## Classification Report\n\n")
    file.write(f"```text\n{class_report}\n```\n")

    file.write(
        "\n## Interpretation\n\n"
        "This model uses pretrained DistilBERT weights with a newly initialized, "
        "untrained classification head. Its predictions are not yet adapted to "
        "the CoLA classification task. This is a reference experiment, not a "
        "meaningful estimate of the best performance DistilBERT can achieve.\n"
    )

print(f"\nModel saved: {model_path}")
print(f"Report saved: {report_path}")