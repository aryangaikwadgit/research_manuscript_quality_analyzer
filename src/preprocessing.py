
import pandas as pd

cols_name = ["sentence", "label"]


class DataPreprocessor:

    # load required columns
    def load_data(self, file_path):
        df = pd.read_csv(file_path, usecols=cols_name)
        return df

    # clean sentences
    def clean_sentences(self, df):
        df = df.dropna(subset=["sentence"]).copy()
        df["sentence"] = df["sentence"].str.strip()
        df = df[df["sentence"] != ""].copy()
        return df

    # remove conflicting duplicates
    def remove_conflicting_duplicates(self, df):
        sentences_to_remove = []

        for sentence in df["sentence"].unique():
            sentence_rows = df[df["sentence"] == sentence]
            labels = sentence_rows["label"].unique()

            if len(labels) > 1:
                sentences_to_remove.append(sentence)

        for sentence in sentences_to_remove:
            df = df[df["sentence"] != sentence]

        return df

    # remove consistent duplicates
    def remove_consistent_duplicates(self, df):
        df = df.drop_duplicates(subset=["sentence"], keep="first")
        return df

    # run preprocessing
    def preprocess(self, file_path):
        df = self.load_data(file_path)
        df = self.clean_sentences(df)
        df = self.remove_conflicting_duplicates(df)
        df = self.remove_consistent_duplicates(df)
        df = df.reset_index(drop=True)
        return df
