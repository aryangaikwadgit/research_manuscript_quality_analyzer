
import joblib


class SentencePredictor:

    # load saved model and vectorizer
    def __init__(self, model_path, vectorizer_path):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    # predict multiple sentences
    def predict(self, sentences):
        features = self.vectorizer.transform(sentences)
        predictions = self.model.predict(features)
        return predictions

    # predict one sentence
    def predict_one(self, sentence):
        prediction = self.predict([sentence])
        return prediction[0]

    
    # get prediction probabilities
    def predict_proba(self, sentences):
        features = self.vectorizer.transform(sentences)
        probabilities = self.model.predict_proba(features)
        return probabilities

