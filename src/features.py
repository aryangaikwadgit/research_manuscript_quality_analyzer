
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtractor:

    # initialize vectorizer
    def __init__(self):
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=50000)

    # fit vectorizer on training data
    def fit_transform(self, sentences):
        features = self.vectorizer.fit_transform(sentences)
        return features

    # transform validation or test data
    def transform(self, sentences):
        features = self.vectorizer.transform(sentences)
        return features
