
from sklearn.linear_model import LogisticRegression


class ModelTrainer:

    # initialize model
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")

    # train model
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model

    # predict labels
    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions
