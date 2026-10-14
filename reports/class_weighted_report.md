# Class-Weighted Model Evaluation Report

## 1. Model Information

- **Project:** Research Manuscript Quality Analyzer
- **Dataset:** CoLA (Corpus of Linguistic Acceptability)
- **Model:** Logistic Regression
- **Feature Extraction:** TF-IDF
- **Class Weighting:** `balanced`
- **Maximum Iterations:** 1000
- **Evaluation Split:** Validation
- **Validation Samples Evaluated:** 1,036

## 2. Evaluation Metrics

| Metric | Score |
|---|---:|
| Accuracy | 0.6274 |
| Precision (Class 1) | 0.7409 |
| Recall (Class 1) | 0.7099 |
| F1-score (Class 1) | 0.7251 |
| Macro F1-score | 0.5700 |
| Weighted F1-score | 0.6300 |

## 3. Confusion Matrix

| Actual / Predicted | Class 0 | Class 1 |
|---|---:|---:|
| Class 0 | 141 | 178 |
| Class 1 | 208 | 509 |

## 4. Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 — Unacceptable | 0.40 | 0.44 | 0.42 | 319 |
| 1 — Acceptable | 0.74 | 0.71 | 0.73 | 717 |
| Macro average | 0.57 | 0.58 | 0.57 | 1,036 |
| Weighted average | 0.64 | 0.63 | 0.63 | 1,036 |

## 5. Comparison with Baseline

- Validation accuracy decreased from 69.69% to 62.74%.
- Class 0 recall increased from 5.02% to 44.20%.
- Class 0 F1-score increased from 0.09 to 0.42.
- Macro F1-score increased from approximately 0.46 to 0.57.
- The model identifies more unacceptable sentences but also misclassifies more acceptable sentences.

## 6. Conclusion

Class weighting improves the balance between performance on the two classes, as reflected by the higher macro F1-score. However, the reduction in accuracy indicates a trade-off between identifying unacceptable sentences and correctly classifying acceptable sentences.

The class-weighted model is a promising candidate for further experimentation, but it is not yet selected as the final model.

## 7. Experiment Status

- [x] Initial baseline evaluated
- [x] Class-weighted experiment evaluated
- [x] Results documented
- [ ] Compare additional model settings
- [ ] Select the final model
- [ ] Evaluate the selected model on the test set

**Status:** Second experiment completed.