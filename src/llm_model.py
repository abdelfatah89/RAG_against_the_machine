from typing import List, Dict, cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline)


class LLModel:
    def __init__(self,
                 model_name: str = "Qwen/Qwen3-0.6B",
                 *,
                 dtype: torch.dtype | None = None,
                 trust_remote_code: bool = True
                 ) -> None:

        self._model_name = model_name
        self._dtype = dtype or torch.float16

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self._dtype,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
        self._model.eval()

        self.generator = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
        )

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a response from the LLM based on the given prompt.
        """
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(prompt, str):
            raise TypeError("Expected string prompt from apply_chat_template")

        result = self.generator(
            prompt,
            do_sample=False,
            return_full_text=False,
        )

        return cast(str, result[0]["generated_text"])

    def generate_prompt(self,
                        question: str,
                        sources: List[str] | None = None
                        ) -> List[Dict[str, str]]:
        """
        Generate a prompt for the LLM based on
        the question and retrieved sources.
        """
        sources = sources or []
        context = "\n\n".join(
                    f"[Context {i + 1}]\n{source}"
                    for i, source in enumerate(sources)
                )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful RAG assistant. Answer the user's"
                    " question using only the provided context. If the answer"
                    " cannot be found in the context, say you don't know."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question:\n{question}"
                ),
            },
        ]

        return messages
