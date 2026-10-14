from src.preprocessing import DataPreprocessor
from src.features import FeatureExtractor
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from src.save_model import save_model

# set project paths
PROJECT_ROOT = (
    r"C:\Main_Drive\work\workarea\projects" r"\research_manuscript_quality_analyzer"
)

train_path = PROJECT_ROOT + r"\data\raw\cola_train.csv"
val_path = PROJECT_ROOT + r"\data\raw\cola_validation.csv"

model_path = PROJECT_ROOT + r"\models\logistic_regression.pkl"
vectorizer_path = PROJECT_ROOT + r"\models\tfidf_vectorizer.pkl"


# initialize objects
preprocessor = DataPreprocessor()
extractor = FeatureExtractor()
trainer = ModelTrainer()
evaluator = ModelEvaluator()


# load and preprocess training data
df_train = preprocessor.preprocess(train_path)

X_train = extractor.fit_transform(df_train["sentence"])
y_train = df_train["label"]


# train the logistic regression model
trainer.train(X_train, y_train)


# save the model and vectorizer
save_model(trainer.model, extractor, model_path, vectorizer_path)


# load and preprocess validation data
df_val = preprocessor.preprocess(val_path)

X_val = extractor.transform(df_val["sentence"])
y_val = df_val["label"]


# evaluate the model
y_pred = trainer.predict(X_val)

evaluator.evaluate(y_val, y_pred)
evaluator.display_results()
