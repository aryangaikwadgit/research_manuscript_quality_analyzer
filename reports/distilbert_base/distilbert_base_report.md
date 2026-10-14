# Pretrained DistilBERT Baseline Report

## Experimental Setup

- Model: DistilBERT base uncased
- Fine-tuning: Not performed
- Classification head: Newly initialized and untrained
- Task: Binary linguistic acceptability classification
- Validation examples: 1043
- Evaluation time (seconds): 5.49

## Validation Metrics

- **eval_loss:** 0.7130
- **eval_model_preparation_time:** 0.0010
- **eval_accuracy:** 0.3087
- **eval_macro_precision:** 0.4042
- **eval_macro_recall:** 0.4991
- **eval_macro_f1:** 0.2369
- **eval_runtime:** 5.4899
- **eval_samples_per_second:** 189.9840
- **eval_steps_per_second:** 12.0220

## Confusion Matrix

Rows = actual labels `[0, 1]`; columns = predicted labels `[0, 1]`.

```text
[[321   1]
 [720   1]]
```

## Classification Report

```text
              precision    recall  f1-score   support

           0       0.31      1.00      0.47       322
           1       0.50      0.00      0.00       721

    accuracy                           0.31      1043
   macro avg       0.40      0.50      0.24      1043
weighted avg       0.44      0.31      0.15      1043

```

## Interpretation

This model uses pretrained DistilBERT weights with a newly initialized, untrained classification head. Its predictions are not yet adapted to the CoLA classification task. This is a reference experiment, not a meaningful estimate of the best performance DistilBERT can achieve.
