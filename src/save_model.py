
import joblib
import os


# save model and vectorizer
def save_model(model, extractor, model_path, vectorizer_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(extractor.vectorizer, vectorizer_path)

    print("Model and vectorizer saved successfully.")


# load model and vectorizer
def load_model(model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer
