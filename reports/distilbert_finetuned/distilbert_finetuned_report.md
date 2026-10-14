# Fine-Tuned DistilBERT Report

## Experimental Setup

- Model: DistilBERT base uncased
- Fine-tuning epochs: 2
- Training batch size: 8
- Learning rate: 2e-5
- Maximum sequence length: 128 tokens
- Evaluation: validation dataset
- Selection metric: macro F1-score
- Training examples: 8521
- Validation examples: 1036
- Training time (seconds): 780.19

## Validation Metrics

- **eval_loss:** 0.6140
- **eval_accuracy:** 0.8012
- **eval_macro_precision:** 0.7768
- **eval_macro_recall:** 0.7345
- **eval_macro_f1:** 0.7491
- **eval_runtime:** 7.0103
- **eval_samples_per_second:** 147.7830
- **eval_steps_per_second:** 9.2720

## Confusion Matrix

Rows = actual labels `[0, 1]`; columns = predicted labels `[0, 1]`.

```text
[[179 140]
 [ 66 651]]
```

## Classification Report

```text
              precision    recall  f1-score   support

unacceptable       0.73      0.56      0.63       319
  acceptable       0.82      0.91      0.86       717

    accuracy                           0.80      1036
   macro avg       0.78      0.73      0.75      1036
weighted avg       0.79      0.80      0.79      1036

```

## Scope and Limitations

This model predicts linguistic acceptability using the CoLA dataset. It does not directly evaluate factual correctness, methodology, citations, originality, or overall academic manuscript quality.
