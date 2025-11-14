from abc import ABC, abstractmethod
from typing import Any


class Strategy(ABC):
    """Prompting strategy."""

    name: str
    """The name of the strategy, should not contain spaces."""

    @abstractmethod
    def generate_response_impl(self, messages: list[dict[str, str]] | str) -> str:
        """Client-specific implementation of the generate_response method."""

    @abstractmethod
    def format(self, sample: str, value: str, all_values: list[str]) -> Any:
        """Format the prompt for the strategy using the sample text and the value.

        All values are provided to allow for more complex prompts.
        """

    def generate_response(self, sample: str, values: list[str]) -> dict[str, str]:
        """Generate a response based on the sample text.

        This returns a dictionary with the generated response to the sample given, according with the implemented strategy, for each value in `values`.
        The dictionary keys are structured as `self.name`_value: response.
        """
        responses = {}
        for v in values:
            key = f"{self.name}_{v}"
            value = self.generate_response_impl(self.format(sample, v, values))
            responses[key] = value
        return responses
