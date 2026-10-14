import os
import pandas as pd
from sklearn.model_selection import train_test_split

# set project paths
PROJECT_ROOT = (
    r"C:\Main_Drive\work\workarea\projects"
    r"\research_manuscript_quality_analyzer"
)

train_path = PROJECT_ROOT + r"\data\raw\cola_train.csv"
output_dir = PROJECT_ROOT + r"\data\processed\final_evaluation"

os.makedirs(output_dir, exist_ok=True)

# load the labeled training data
df = pd.read_csv(train_path, usecols=["sentence", "label"])
df = df.dropna(subset=["sentence", "label"]).copy()
df["sentence"] = df["sentence"].str.strip()
df = df[df["sentence"] != ""].copy()
df = df[df["label"].isin([0, 1])].copy()
df["label"] = df["label"].astype(int)

# remove conflicting and repeated sentences
label_counts = df.groupby("sentence")["label"].nunique()
conflicting = label_counts[label_counts > 1].index
df = df[~df["sentence"].isin(conflicting)].copy()
df = df.drop_duplicates(subset=["sentence"], keep="first")
df = df.reset_index(drop=True)

# create a stratified holdout
development_df, final_df = train_test_split(
    df,
    test_size=0.15,
    random_state=42,
    stratify=df["label"]
)

# save separate datasets without overwriting existing data
development_path = output_dir + r"\development.csv"
final_path = output_dir + r"\final_holdout.csv"

development_df.to_csv(development_path, index=False)
final_df.to_csv(final_path, index=False)

print(f"Development examples: {len(development_df)}")
print(f"Final holdout examples: {len(final_df)}")
print("\nDevelopment label distribution:")
print(development_df["label"].value_counts().sort_index())
print("\nHoldout label distribution:")
print(final_df["label"].value_counts().sort_index())
print(f"\nDevelopment data saved: {development_path}")
print(f"Final holdout saved: {final_path}")
