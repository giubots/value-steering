from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from value_steering.evaluators.Evaluator import Evaluator


class Schwartz(Evaluator):
    """Evaluator using Schwartz from https://huggingface.co/devnote5676/schwartz-values-classifier."""

    def __init__(self):
        self.name = "schwartz"
        self.values = [
            "achievement",
            "benevolence",
            "conformity",
            "hedonism",
            "power",
            "security",
            "self-direction",
            "stimulation",
            "tradition",
            "universalism",
        ]
        cache_dir = "./.model_cache"
        self.tokenizer = AutoTokenizer.from_pretrained(
            "devnote5676/schwartz-values-classifier",
            cache_dir=cache_dir,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "devnote5676/schwartz-values-classifier",
            cache_dir=cache_dir,
            device_map="auto",
        )
        self.pipe = pipeline(
            "text-classification",
            model="devnote5676/schwartz-values-classifier",
        )

    def get_scores(self, prefix: str, sentence: str) -> dict[str, int]:
        scores = {}
        for v in self.values:
            key = f"{self.name}_{prefix}_{v}"
            output = self.pipe(f"<{v}>[SEP]{sentence}")
            scores[key] = output
        return scores
