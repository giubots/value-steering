from openai import OpenAI

from value_steering.strategies.Strategy import Strategy


class GPTBase(Strategy):
    _client: OpenAI
    _model: str

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.name = "gptbase"
        self._client = OpenAI()
        self._model = model

    def generate_response_impl(self, messages: list[dict[str, str]] | str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content

    def format(
        self,
        sample: str,
        value: str,
        all_values: list[str],
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "You are having a conversation."},
            {"role": "user", "content": sample},
            {
                "role": "user",
                "content": "Generate a short response.",
            },
        ]


class GPTSimple(GPTBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "gptsimple"

    def format(
        self,
        sample: str,
        value: str,
        all_values: list[str],
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "You are having a conversation, your responses are based on a specific value.",
            },
            {"role": "user", "content": sample},
            {
                "role": "user",
                "content": f"Generate a short response that aligns with the value '{value}'.",
            },
        ]


class GPTAdvanced(GPTBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "gptadv"

    def format(
        self,
        sample: str,
        value: str,
        all_values: list[str],
    ) -> list[dict[str, str]]:
        values = ", ".join(all_values)
        system = f"You are having a conversation, your responses are based on a specific value from the Moral Foundation Theory. The values work in pairs: for example, care is the opposite of harm. The possible values are: {values}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": sample},
            {
                "role": "user",
                "content": f"Generate a short response that maximises the value '{value}', or minimises its opposite.",
            },
        ]
