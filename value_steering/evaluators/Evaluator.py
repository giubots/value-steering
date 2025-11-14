from abc import ABC, abstractmethod


class Evaluator(ABC):
    """Extracts values from text."""

    name: str
    """The name of the evaluator, should not contain spaces."""
    values: list[str]
    """The values that the evaluator can score."""

    @abstractmethod
    def get_scores(self, prefix: str, sentence: str) -> dict[str, int]:
        """Get scores for the given sentence.

        This returns a dictionary with the scores for each value in `self.values`.
        The dictionary keys are structured as `self.name`_`prefix`_value: score.
        """
