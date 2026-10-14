from src.predict_distilbert import DistilBertPredictor

MODEL_PATH = (
    r"models\distilbert_weighted_lr1e5\run_20261013_160858"
)

predictor = DistilBertPredictor(MODEL_PATH)

test_sentences = [
    "The scientist conducted the experiment carefully.",
    "The scientist the experiment carefully conducted.",
]

for sentence in test_sentences:
    result = predictor.predict_one(sentence)

    print("\nSentence:", sentence)
    print("Prediction:", result["prediction"])
    print(
        "Unacceptable:",
        f'{result["probabilities"]["Unacceptable"]:.2%}'
    )
    print(
        "Acceptable:",
        f'{result["probabilities"]["Acceptable"]:.2%}'
    )
