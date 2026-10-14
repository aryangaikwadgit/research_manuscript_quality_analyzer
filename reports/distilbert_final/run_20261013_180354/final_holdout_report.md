# DistilBERT Final Holdout Evaluation

- Experiment ID: `20261013_180354`
- Model: DistilBERT base uncased
- Training: class-weighted cross-entropy
- Learning rate: `1e-5`
- Maximum epochs: 4
- Training batch size: 8
- Maximum sequence length: 128 tokens
- Training examples: 6155
- Internal validation examples: 1087
- Final holdout examples: 1279
- Training time: 1100.70 seconds

## Final Holdout Metrics

- Accuracy: 0.7873
- Macro precision: 0.7528
- Macro recall: 0.7055
- Macro F1: 0.7206

## Confusion Matrix

Rows = actual labels `[0, 1]`; columns = predicted labels `[0, 1]`.

```text
[[191 187]
 [ 85 816]]
```

## Classification Report

```text
              precision    recall  f1-score   support

unacceptable       0.69      0.51      0.58       378
  acceptable       0.81      0.91      0.86       901

    accuracy                           0.79      1279
   macro avg       0.75      0.71      0.72      1279
weighted avg       0.78      0.79      0.78      1279

```

## Evaluation Notes

The final holdout was separated before this experiment. The model was selected using an internal validation subset of the development data. This is a held-out evaluation of a new stratified split from CoLA, not the official CoLA test set. Results measure linguistic acceptability, not complete academic manuscript quality, factual correctness, citations, or originality.
