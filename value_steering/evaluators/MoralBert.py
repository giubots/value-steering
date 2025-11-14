import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoModel, AutoTokenizer

from value_steering.evaluators.Evaluator import Evaluator


class MoralBertUnit(
    nn.Module,
    PyTorchModelHubMixin,
    pipeline_tag="text-classification",
    license="mit",
):
    def __init__(self, bert_model, moral_label=2):

        super(MoralBertUnit, self).__init__()
        self.bert = bert_model
        bert_dim = 768
        self.invariant_trans = nn.Linear(768, 768)
        self.moral_classification = nn.Sequential(
            nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, moral_label)
        )

    def forward(self, input_ids, token_type_ids, attention_mask):
        pooled_output = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]

        pooled_output = self.invariant_trans(pooled_output)

        logits = self.moral_classification(pooled_output)

        return logits


class MoralBert(Evaluator):
    """Evaluator using MoralBert from https://doi.org/10.1145/3677525.3678694."""

    _units: dict

    def __init__(self):
        self.name = "moralbert"
        self.values = [
            "care",
            "harm",
            "fairness",
            "cheating",
            "loyalty",
            "betrayal",
            "authority",
            "subversion",
            "purity",
            "degradation",
        ]
        cache_dir = "./.model_cache"

        # BERT model and tokenizer:
        self.bert_model = AutoModel.from_pretrained(
            "bert-base-uncased",
            cache_dir=cache_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            cache_dir=cache_dir,
        )

        self._units = {}
        for value in self.values:
            repo_name = f"vjosap/moralBERT-predict-{value}-in-text"
            self._units[value] = MoralBertUnit.from_pretrained(
                repo_name,
                bert_model=self.bert_model,
                cache_dir=cache_dir,
            )

    def preprocessing(self, input_text):
        """
        Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
        - input_ids: list of token ids
        - token_type_ids: list of token type ids
        - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
        """
        return self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=150,
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="pt",
            truncation=True,
        )

    def get_scores(self, prefix: str, sentence: str) -> dict[str, int]:
        scores = {}
        encoded = self.preprocessing(sentence)
        for mft, model in self._units.items():
            output = model(**encoded)
            score = F.softmax(output, dim=1)
            mft_value = score[0, 1].item()
            key = f"{self.name}_{prefix}_{mft}"
            scores[key] = mft_value
        return scores
