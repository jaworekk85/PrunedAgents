from __future__ import annotations

from role_pruning.models.base import Generation


class HFCausalLMBackend:
    """Thin Hugging Face entry point. Real activation hooks live in activations/hooks.py."""

    def __init__(
        self,
        model_id: str,
        revision: str | None,
        device: str,
        dtype: str,
        max_new_tokens: int,
        quantization: str = "none",
        use_chat_template: bool = False,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Hugging Face backend requires torch and transformers. "
                "Install project dependencies before using backend='hf'."
            ) from exc

        self.device = _resolve_device(device, torch)
        torch_dtype = _resolve_dtype(dtype, self.device, torch)
        load_kwargs = {}
        if torch_dtype is not None:
            load_kwargs["dtype"] = torch_dtype
        if quantization in {"int8", "int4"}:
            try:
                from transformers import BitsAndBytesConfig  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    f"{quantization} quantization requires bitsandbytes. "
                    "Install the project's GPU dependencies first."
                ) from exc
            if quantization == "int8":
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
            load_kwargs["device_map"] = "auto"
        elif quantization != "none":
            raise ValueError(f"Unsupported Hugging Face quantization mode: {quantization!r}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            low_cpu_mem_usage=True,
            **load_kwargs,
        )
        if quantization == "none":
            self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.model_id = model_id
        self.revision = revision
        self.dtype = str(torch_dtype or "model_default")
        self.quantization = quantization
        self.use_chat_template = use_chat_template

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    def generate(
        self, prompt: str, role: str, task: dict[str, str], *, system: str | None = None
    ) -> Generation:
        import torch

        inputs = self.tokenize_prompt(prompt, system=system)
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return Generation(text=text, generated_token_count=int(generated.numel()))

    def tokenize_prompt(self, prompt: str, *, system: str | None = None):
        if not self.use_chat_template:
            return self.tokenizer(_raw_with_system(prompt, system), return_tensors="pt")
        return self.tokenizer.apply_chat_template(
            _messages(prompt, system),
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

    def tokenize_conversation(self, prompt: str, response: str, *, system: str | None = None):
        if not self.use_chat_template:
            return self.tokenizer(_raw_with_system(prompt, system) + response, return_tensors="pt")
        return self.tokenizer.apply_chat_template(
            [
                *_messages(prompt, system),
                {"role": "assistant", "content": response},
            ],
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )


def _raw_with_system(prompt: str, system: str | None) -> str:
    return f"{system}\n\n{prompt}" if system else prompt


def _messages(prompt: str, system: str | None) -> list[dict[str, str]]:
    user = [{"role": "user", "content": prompt}]
    return [{"role": "system", "content": system}, *user] if system else user


def _resolve_device(device: str, torch_module) -> str:
    if device == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    return device


def _resolve_dtype(dtype: str, device: str, torch_module):
    if dtype == "auto":
        return torch_module.float16 if device == "cuda" else torch_module.float32
    if dtype in {"float16", "fp16"}:
        return torch_module.float16
    if dtype in {"bfloat16", "bf16"}:
        return torch_module.bfloat16
    if dtype in {"float32", "fp32"}:
        return torch_module.float32
    return None
