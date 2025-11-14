import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from value_steering.evaluators.Evaluator import Evaluator


class DeBERTa(Evaluator):
    """Evaluator using Schwartz from https://huggingface.co/nharrel/Valuesnet_DeBERTa_v3"""

    def __init__(self):
        self.name = "deberta"
        self.values = [
            "benevolence",
            "universalism",
            "self-direction",
            "stimulation",
            "hedonism",
            "achievement",
            "power",
            "security",
            "conformity",
            "tradition",
        ]
        cache_dir = "./.model_cache"

        # Load the model and tokenizer
        model_path = "nharrel/Valuesnet_DeBERTa_v3"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=cache_dir,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            cache_dir=cache_dir,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)
        self.model.eval()
        # Define maximum length for padding and truncation
        self.max_length = 128

    def custom_round(self, x):
        if x >= 0.50:
            return 1
        elif x < -0.50:
            return -1
        else:
            return 0

    def predict(self, text):
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)

        prediction = torch.tanh(outputs.logits).cpu().numpy()
        rounded_prediction = self.custom_round(prediction[0][0])
        return rounded_prediction

    def get_scores(self, prefix: str, sentence: str) -> dict[str, int]:
        scores = {}
        for v in self.values:
            output = self.predict(f"[{v.upper()}] {sentence}")
            key = f"{self.name}_{prefix}_{v}"
            scores[key] = output
        return scores
