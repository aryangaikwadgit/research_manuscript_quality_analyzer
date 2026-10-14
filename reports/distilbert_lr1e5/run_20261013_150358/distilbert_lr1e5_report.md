# DistilBERT Learning Rate Experiment

- Experiment ID: `20261013_150358`
- Model: DistilBERT base uncased
- Learning rate: `1e-5`
- Maximum epochs: 3
- Training batch size: 8
- Weight decay: 0.01
- Maximum sequence length: 128 tokens
- Best checkpoint selected by validation macro F1
- Training examples: 8521
- Validation examples: 1036
- Training time: 1134.08 seconds

## Validation Metrics

- **eval_loss:** 0.6256
- **eval_accuracy:** 0.7973
- **eval_macro_precision:** 0.7761
- **eval_macro_recall:** 0.7230
- **eval_macro_f1:** 0.7395
- **eval_runtime:** 6.6252
- **eval_samples_per_second:** 156.3720
- **eval_steps_per_second:** 9.8110

## Confusion Matrix

Rows = actual labels `[0, 1]`; columns = predicted labels `[0, 1]`.

```text
[[169 150]
 [ 60 657]]
```

## Classification Report

```text
              precision    recall  f1-score   support

unacceptable       0.74      0.53      0.62       319
  acceptable       0.81      0.92      0.86       717

    accuracy                           0.80      1036
   macro avg       0.78      0.72      0.74      1036
weighted avg       0.79      0.80      0.79      1036

```

## Limitations

This experiment predicts linguistic acceptability using the CoLA dataset. It does not directly evaluate factual correctness, research methodology, citation quality, originality, or overall academic manuscript quality.
