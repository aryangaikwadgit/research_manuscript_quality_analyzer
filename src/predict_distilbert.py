
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class DistilBertPredictor:

    # load the saved model and tokenizer
    def __init__(self, model_path):
        self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )

        self.model.to(self.device)
        self.model.eval()

    # predict one sentence and return its probabilities
    def predict_one(self, sentence):
        if not isinstance(sentence, str) or not sentence.strip():
            raise ValueError("Please provide a non-empty sentence.")

        inputs = self.tokenizer(
            sentence.strip(),
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)[0]

        predicted_label = int(torch.argmax(probabilities).item())

        labels = {
            0: "Unacceptable",
            1: "Acceptable"
        }

        return {
            "label": predicted_label,
            "prediction": labels[predicted_label],
            "probabilities": {
                "Unacceptable": float(probabilities[0].item()),
                "Acceptable": float(probabilities[1].item())
            }
        }
