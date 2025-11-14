from llama_cpp import Llama

from value_steering.strategies.Strategy import Strategy


class VicunaBase(Strategy):
    _llm: Llama

    def __init__(self):
        self.name = "vicunabase"
        self._llm = Llama.from_pretrained(
            repo_id="TheBloke/Wizard-Vicuna-13B-Uncensored-GGUF",
            filename="Wizard-Vicuna-13B-Uncensored.Q4_K_M.gguf",
            local_dir=".model_cache/llama",
            cache_dir=".model_cache",
            n_gpu_layers=100,
        )

    def generate_response_impl(self, messages: list[dict[str, str]] | str) -> str:
        response = self._llm(
            prompt=messages,
            temperature=0,
            max_tokens=256,
            stop=["USER:", "ASSISTANT:"],
        )
        return response["choices"][0]["text"].strip()

    def format(
        self,
        sample: str,
        value: str,
        all_values: list[str],
    ) -> str:
        return f"You are having a conversation.\n\nUSER: {sample.strip()}\nUSER: Generate a short response.\nASSISTANT:"


class VicunaSimple(VicunaBase):

    def __init__(self):
        super().__init__()
        self.name = "vicunasimple"

    def format(
        self,
        sample: str,
        value: str,
        all_values: list[str],
    ) -> str:
        return f"You are having a conversation, your responses are based on a specific value.\n\nUSER: {sample.strip()}\nUSER: Generate a short response that aligns with the value '{value}'.\nASSISTANT:"


class VicunaAdvanced(VicunaBase):

    def __init__(self):
        super().__init__()
        self.name = "vicunaadvanced"
        # self.llm = Llama.from_pretrained(
        #     repo_id="TheBloke/Wizard-Vicuna-13B-Uncensored-GGUF",
        #     filename="Wizard-Vicuna-13B-Uncensored.Q4_K_M.gguf",
        #     local_dir=".model_cache/llama",
        #     cache_dir=".model_cache",
        #     n_gpu_layers=100,
        #     n_ctx=1024,
        # )

    def format(
        self,
        sample: str,
        value: str,
        all_values: list[str],
    ) -> str:
        values = ", ".join(all_values)
        system = f"You are having a conversation, your responses are based on a specific value from the Moral Foundation Theory. The values work in pairs: for example, care is the opposite of harm. The possible values are: {values}"
        return f"{system}\n\nUSER: {sample.strip()}\nUSER: Generate a short, relevant response that maximises the value '{value}' or minimises its opposite.\nASSISTANT:"
