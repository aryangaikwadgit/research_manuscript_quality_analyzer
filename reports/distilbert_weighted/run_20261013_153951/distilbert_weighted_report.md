# Class-Weighted DistilBERT Experiment

- Experiment ID: `20261013_153951`
- Model: DistilBERT base uncased
- Loss: class-weighted cross-entropy
- Learning rate: `2e-5`
- Maximum epochs: 3
- Training batch size: 8
- Weight decay: 0.01
- Maximum sequence length: 128 tokens
- Early stopping: patience 1, based on validation macro F1
- Training examples: 8521
- Validation examples: 1036
- Training time: 1195.26 seconds

## Class Weights

- Unacceptable (0): 1.6934
- Acceptable (1): 0.7095

## Validation Metrics

- **eval_loss:** 0.9644
- **eval_accuracy:** 0.8060
- **eval_macro_precision:** 0.7816
- **eval_macro_recall:** 0.7432
- **eval_macro_f1:** 0.7570
- **eval_runtime:** 6.7567
- **eval_samples_per_second:** 153.3290
- **eval_steps_per_second:** 9.6200

## Confusion Matrix

Rows = actual labels `[0, 1]`; columns = predicted labels `[0, 1]`.

```text
[[185 134]
 [ 67 650]]
```

## Classification Report

```text
              precision    recall  f1-score   support

unacceptable       0.73      0.58      0.65       319
  acceptable       0.83      0.91      0.87       717

    accuracy                           0.81      1036
   macro avg       0.78      0.74      0.76      1036
weighted avg       0.80      0.81      0.80      1036

```

## Limitations

This model predicts linguistic acceptability using the CoLA dataset. It does not directly assess factual correctness, research methodology, citation quality, originality, or overall manuscript quality. Class weighting may improve minority-class recall but can reduce accuracy or precision for the other class.
