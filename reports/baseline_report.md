
# Baseline Model Evaluation Report

## 1. Model Information

- **Project:** Research Manuscript Quality Analyzer
- **Dataset:** CoLA (Corpus of Linguistic Acceptability)
- **Model:** Logistic Regression
- **Feature Extraction:** TF-IDF
- **Class Weighting:** None
- **Maximum Iterations:** 1000
- **Evaluation Split:** Validation
- **Validation Samples Evaluated:** 1,036

## 2. Evaluation Metrics

| Metric | Score |
|---|---:|
| Accuracy | 0.6969 |
| Precision (Class 1) | 0.6997 |
| Recall (Class 1) | 0.9847 |
| F1-score (Class 1) | 0.8181 |
| Macro F1-score | 0.4600 |
| Weighted F1-score | 0.5900 |

## 3. Confusion Matrix

| Actual / Predicted | Class 0 | Class 1 |
|---|---:|---:|
| Class 0 | 16 | 303 |
| Class 1 | 11 | 706 |

**Interpretation:**
- **True negatives:** 16
- **False positives:** 303
- **False negatives:** 11
- **True positives:** 706

## 4. Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 — Unacceptable | 0.59 | 0.05 | 0.09 | 319 |
| 1 — Acceptable | 0.70 | 0.98 | 0.82 | 717 |
| Macro average | 0.65 | 0.52 | 0.46 | 1,036 |
| Weighted average | 0.67 | 0.70 | 0.59 | 1,036 |

## 5. Observations

- The model achieved **69.69% validation accuracy**.
- Recall for class `1` (acceptable sentences) is high at **98.47%**.
- Recall for class `0` (unacceptable sentences) is very low at approximately **5.02%**.
- The model correctly identified only 16 of the 319 unacceptable sentences.
- The model incorrectly classified 303 unacceptable sentences as acceptable.
- The difference in class-wise recall indicates that the model performs unevenly across the two classes.

## 6. Limitations

- The dataset has an imbalanced class distribution.
- Accuracy alone does not adequately describe the model's performance.
- CoLA evaluates linguistic acceptability rather than overall academic manuscript quality.
- These results represent the initial baseline and should not be treated as final project performance.

## 7. Next Experiment

Train Logistic Regression with `class_weight="balanced"` and compare its performance against this baseline.

The comparison should include:

- Accuracy
- Precision and recall for both classes
- Macro F1-score
- Confusion matrix

The objective is to improve detection of unacceptable sentences without unnecessarily sacrificing performance on acceptable sentences.

## 8. Experiment Status

- [x] Initial preprocessing pipeline implemented
- [x] TF-IDF feature extraction implemented
- [x] Logistic Regression baseline trained
- [x] Validation evaluation completed
- [x] Baseline results documented
- [ ] Class-weighted experiment
- [ ] Model comparison
- [ ] Final test-set evaluation

**Status:** Initial baseline recorded.

