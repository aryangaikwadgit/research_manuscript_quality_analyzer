import os
import time
import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from preprocessing import DataPreprocessor


# set project paths
PROJECT_ROOT = r"C:\Main_Drive\work\workarea\projects\research_manuscript_quality_analyzer"

train_path = PROJECT_ROOT + r"\data\raw\cola_train.csv"
val_path = PROJECT_ROOT + r"\data\raw\cola_validation.csv"
model_dir = PROJECT_ROOT + r"\models"
report_dir = PROJECT_ROOT + r"\reports"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)


# evaluate and save experiment results
def run_experiment(name, X_train, X_val, y_train, y_val, vectorizers):
    model = LinearSVC(class_weight="balanced", random_state=42)

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    predictions = model.predict(X_val)

    results = {
        "accuracy": accuracy_score(y_val, predictions),
        "macro_precision": precision_score(
            y_val, predictions, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_val, predictions, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_val, predictions, average="macro", zero_division=0
        ),
        "training_time_seconds": round(training_time, 2),
        "confusion_matrix": confusion_matrix(y_val, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_val, predictions, labels=[0, 1], zero_division=0
        )
    }

    model_path = model_dir + "\\" + name + "_model.pkl"
    vectorizer_path = model_dir + "\\" + name + "_vectorizers.pkl"
    report_path = report_dir + "\\" + name + "_report.md"

    joblib.dump(model, model_path)
    joblib.dump(vectorizers, vectorizer_path)

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(f"# {name.replace('_', ' ').title()} Evaluation Report\n\n")
        file.write("## Validation Metrics\n\n")
        for metric, value in results.items():
            if metric not in ["confusion_matrix", "classification_report"]:
                file.write(f"- **{metric.replace('_', ' ').title()}:** {value}\n")

        file.write("\n## Confusion Matrix\n\n")
        file.write("Rows = actual labels `[0, 1]`; columns = predicted labels `[0, 1]`.\n\n")
        file.write(f"```text\n{results['confusion_matrix']}\n```\n")
        file.write("\n## Classification Report\n\n")
        file.write(f"```text\n{results['classification_report']}\n```\n")

    print(f"\n{'=' * 50}")
    print(name)
    print(f"{'=' * 50}")
    for metric, value in results.items():
        print(f"\n{metric}:")
        print(value)

    print(f"\nModel saved: {model_path}")
    print(f"Vectorizer saved: {vectorizer_path}")
    print(f"Report saved: {report_path}")

    return results


# load and preprocess training and validation data
preprocessor = DataPreprocessor()
df_train = preprocessor.preprocess(train_path)
df_val = preprocessor.preprocess(val_path)

X_train_text = df_train["sentence"]
y_train = df_train["label"]
X_val_text = df_val["sentence"]
y_val = df_val["label"]


# experiment 1: word-level tf-idf with linear svm
word_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    max_features=50000
)

X_train_word = word_vectorizer.fit_transform(X_train_text)
X_val_word = word_vectorizer.transform(X_val_text)

word_results = run_experiment(
    "svm_word_tfidf",
    X_train_word,
    X_val_word,
    y_train,
    y_val,
    {"word": word_vectorizer}
)


# experiment 2: combine word and character tf-idf
word_vectorizer_combined = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    max_features=50000
)

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=50000,
    min_df=2
)

X_train_word_combined = word_vectorizer_combined.fit_transform(X_train_text)
X_val_word_combined = word_vectorizer_combined.transform(X_val_text)

X_train_char = char_vectorizer.fit_transform(X_train_text)
X_val_char = char_vectorizer.transform(X_val_text)

X_train_combined = hstack([X_train_word_combined, X_train_char])
X_val_combined = hstack([X_val_word_combined, X_val_char])

combined_results = run_experiment(
    "svm_word_char_tfidf",
    X_train_combined,
    X_val_combined,
    y_train,
    y_val,
    {
        "word": word_vectorizer_combined,
        "char": char_vectorizer
    }
)


# compare the two experiments
comparison = pd.DataFrame([
    {"model": "Word TF-IDF + Linear SVM", **word_results},
    {"model": "Word + Character TF-IDF + Linear SVM", **combined_results}
])

comparison_path = report_dir + r"\svm_comparison.csv"
comparison.to_csv(comparison_path, index=False)

print("\nExperiment comparison:")
print(comparison[[
    "model", "accuracy", "macro_precision",
    "macro_recall", "macro_f1", "training_time_seconds"
]].to_string(index=False))

print(f"\nComparison saved: {comparison_path}")