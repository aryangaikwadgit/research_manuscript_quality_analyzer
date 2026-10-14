
from src.predict import SentencePredictor

PROJECT_ROOT = r"C:\Main_Drive\work\workarea\projects\research_manuscript_quality_analyzer"

model_path = PROJECT_ROOT + r"\models\logistic_regression.pkl"
vectorizer_path = PROJECT_ROOT + r"\models\tfidf_vectorizer.pkl"

predictor = SentencePredictor(model_path, vectorizer_path)

sentences = [
    "The researcher conducted the experiment.",
    "The researcher experiment conducted the."
]

predictions = predictor.predict(sentences)
probabilities = predictor.predict_proba(sentences)

for sentence, prediction, probability in zip(sentences, predictions, probabilities):
    label = "Acceptable" if prediction == 1 else "Unacceptable"
    print(f"Sentence: {sentence}")
    print(f"Prediction: {label}")
    print(f"Unacceptable probability: {probability[0]:.2%}")
    print(f"Acceptable probability: {probability[1]:.2%}\n")
