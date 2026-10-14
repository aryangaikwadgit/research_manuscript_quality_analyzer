# Class-Weighted DistilBERT with Learning Rate 1e-5

- Experiment ID: `20261013_160858`
- Model: DistilBERT base uncased
- Loss: balanced class-weighted cross-entropy
- Learning rate: `1e-5`
- Maximum epochs: 4
- Training batch size: 8
- Maximum sequence length: 128 tokens
- Early stopping: patience 1 based on validation macro F1
- Training examples: 8521
- Validation examples: 1036
- Training time: 3722.09 seconds

## Class Weights

- Unacceptable (0): 1.6934
- Acceptable (1): 0.7095

## Validation Metrics

- **eval_loss:** 0.9452
- **eval_accuracy:** 0.8118
- **eval_macro_precision:** 0.7852
- **eval_macro_recall:** 0.7579
- **eval_macro_f1:** 0.7687
- **eval_runtime:** 6.6899
- **eval_samples_per_second:** 154.8600
- **eval_steps_per_second:** 9.7160

## Confusion Matrix

Rows = actual labels `[0, 1]`; columns = predicted labels `[0, 1]`.

```text
[[197 122]
 [ 73 644]]
```

## Classification Report

```text
              precision    recall  f1-score   support

unacceptable       0.73      0.62      0.67       319
  acceptable       0.84      0.90      0.87       717

    accuracy                           0.81      1036
   macro avg       0.79      0.76      0.77      1036
weighted avg       0.81      0.81      0.81      1036

```

## Limitations

This model evaluates linguistic acceptability on CoLA. It does not directly assess factual correctness, methodology, citation quality, originality, or overall manuscript quality.
