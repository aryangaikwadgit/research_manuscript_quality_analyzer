# SVM Model and Vectorization Comparison

## 1. Objective

This experiment evaluates two Linear Support Vector Machine (Linear SVM) configurations for linguistic acceptability classification using the CoLA dataset.

The objective is to compare word-level TF-IDF features against a combination of word-level and character-level TF-IDF features.

## 2. Experimental Setup

- **Dataset:** CoLA
- **Training data:** Cleaned CoLA training split
- **Validation data:** Cleaned validation split containing 1,036 sentences
- **Models:** `LinearSVC`
- **Class weighting:** `balanced`
- **Word TF-IDF:** Lowercasing enabled, word n-grams `(1, 2)`, maximum 50,000 features
- **Character TF-IDF:** Character n-grams `(3, 5)`, maximum 50,000 features, `min_df=2`
- **Evaluation metrics:** Accuracy, macro precision, macro recall, macro F1, confusion matrix, and class-wise classification report

Both experiments used the same preprocessed training and validation datasets.

## 3. Results

| Metric | Word TF-IDF + Linear SVM | Word + Character TF-IDF + Linear SVM |
|---|---:|---:|
| Accuracy | 0.6419 | 0.6429 |
| Macro Precision | 0.5717 | 0.5751 |
| Macro Recall | 0.5681 | 0.5723 |
| Macro F1 | 0.5694 | 0.5734 |
| Training Time (seconds) | 0.06 | 0.28 |

## 4. Confusion Matrices

### Word TF-IDF + Linear SVM

| Actual / Predicted | Class 0 | Class 1 |
|---|---:|---:|
| Class 0 | 120 | 199 |
| Class 1 | 172 | 545 |

### Word + Character TF-IDF + Linear SVM

| Actual / Predicted | Class 0 | Class 1 |
|---|---:|---:|
| Class 0 | 124 | 195 |
| Class 1 | 175 | 542 |

Class 0 represents linguistically unacceptable sentences, while class 1 represents acceptable sentences.

## 5. Analysis

The word + character TF-IDF configuration achieved a macro F1 score of 0.5734, compared with 0.5694 for word-only TF-IDF.

This represents an absolute improvement of approximately 0.40 percentage points in macro F1. Accuracy also increased slightly, from 64.19% to 64.29%.

The combined feature representation correctly classified four additional class 0 examples, but correctly classified three fewer class 1 examples. Overall, the improvement was modest.

Character n-grams can capture subword patterns and word-form differences that word-level features may miss. However, these results do not establish that character features will improve performance on other datasets.

## 6. Limitations

- Both experiments were evaluated on one validation split.
- The observed performance difference is small and has not been tested for statistical significance.
- The dataset has an imbalanced class distribution, so accuracy alone is insufficient.
- CoLA measures linguistic acceptability, not overall research manuscript quality.
- These results should not be interpreted as a complete assessment of grammar, academic writing, scientific validity, or manuscript readiness.

## 7. Conclusion

The word + character TF-IDF configuration is the stronger of the two SVM experiments based on the observed validation metrics. It will be retained as the preferred SVM configuration for subsequent comparisons.

The gain is modest, so the final model should be selected only after considering results from other candidate models, including the existing Logistic Regression baseline and any subsequent neural-model experiment.

## 8. Saved Artifacts

- `models/svm_word_tfidf_model.pkl`
- `models/svm_word_tfidf_vectorizers.pkl`
- `models/svm_word_char_tfidf_model.pkl`
- `models/svm_word_char_tfidf_vectorizers.pkl`
- `reports/svm_word_tfidf_report.md`
- `reports/svm_word_char_tfidf_report.md`
- `reports/svm_comparison.csv`
