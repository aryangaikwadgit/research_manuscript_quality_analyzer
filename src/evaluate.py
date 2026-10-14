
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


class ModelEvaluator:

    # initialize evaluator
    def __init__(self):
        self.results = {}

    # calculate evaluation metrics
    def evaluate(self, y_true, y_pred):
        self.results["accuracy"] = accuracy_score(y_true, y_pred)
        self.results["precision"] = precision_score(y_true, y_pred, zero_division=0)
        self.results["recall"] = recall_score(y_true, y_pred, zero_division=0)
        self.results["f1_score"] = f1_score(y_true, y_pred, zero_division=0)
        self.results["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1])
        self.results["classification_report"] = classification_report(y_true, y_pred, labels=[0, 1], zero_division=0)

        return self.results

    # display evaluation results
    def display_results(self):
        for metric, value in self.results.items():
            print(f"\n{metric}:")
            print(value)
