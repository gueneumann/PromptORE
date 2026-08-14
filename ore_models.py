"""Model loading and prompt/embedding adaptation for PromptORE.

Two model families are supported, each with a different strategy for turning
a prompt into a single relation embedding:

- encoder_mlm (BERT, RoBERTa, ModernBERT, ...): the prompt contains a mask
  token; the embedding is the final hidden state at the mask position.
- causal_lm (OLMo, Qwen, GPT-2, ...): the prompt is a completion-style
  sentence with nothing after the point where the relation word would be
  generated; the embedding is the final hidden state at the last real token.
"""
from abc import ABC, abstractmethod

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_MASKED_LM_MAPPING_NAMES,
)

DEFAULT_TEMPLATES = {
    "encoder_mlm": "{sent} {e1} {mask} {e2}",
    "causal_lm": "{sent} The relation between {e1} and {e2} is",
}


def detect_model_type(model_name: str, override: str = None) -> str:
    """Resolve a HF model name/path to 'encoder_mlm' or 'causal_lm'.

    Args:
        model_name (str): HF model name or local path.
        override (str, optional): explicit 'encoder_mlm'/'causal_lm', skips
            auto-detection when provided.

    Returns:
        str: 'encoder_mlm' or 'causal_lm'
    """
    if override:
        if override not in ("encoder_mlm", "causal_lm"):
            raise ValueError(f"Unknown model_type override: {override!r}; "
                              f"expected 'encoder_mlm' or 'causal_lm'")
        return override

    config = AutoConfig.from_pretrained(model_name)
    # Prefer the checkpoint's own declared architecture: several encoder
    # model_types (bert, roberta, ...) are ALSO registered in the causal-LM
    # mapping (they support an optional decoder variant for encoder-decoder
    # setups), so checking the model_type registries first would misclassify
    # e.g. roberta-large as causal_lm. The saved architecture reflects what
    # this specific checkpoint actually is.
    architectures = getattr(config, "architectures", None) or []
    if any(arch.endswith("ForMaskedLM") for arch in architectures):
        return "encoder_mlm"
    if any(arch.endswith("ForCausalLM") or arch.endswith("LMHeadModel") for arch in architectures):
        return "causal_lm"

    # Fallback for checkpoints without an `architectures` field: use the same
    # model_type -> architecture registries AutoModelForCausalLM/
    # AutoModelForMaskedLM dispatch on. Check masked-LM first for the same
    # reason as above.
    if config.model_type in MODEL_FOR_MASKED_LM_MAPPING_NAMES:
        return "encoder_mlm"
    if config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        return "causal_lm"
    raise ValueError(
        f"Could not auto-detect model family for '{model_name}' "
        f"(model_type={config.model_type!r}, architectures={architectures}). "
        f"Pass --model-type explicitly (encoder_mlm or causal_lm).")


class BaseOreModel(ABC):
    """Common interface for encoder-MLM and causal-LM prompt models."""

    model_type: str

    def __init__(self, model_name: str, device: str = None):
        """
        Args:
            model_name (str): HF model name or local path.
            device (str, optional): force a specific device (e.g. 'cpu',
                'cuda:0'). Defaults to None, which lets accelerate shard the
                model across all visible GPUs via device_map='auto'.
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load()

    @abstractmethod
    def _load(self) -> None:
        """Load self.tokenizer and self.model."""

    def _device_map_arg(self):
        return "auto" if self.device is None else {"": self.device}

    @property
    def input_device(self):
        """Device to place input tensors on. With device_map='auto'-sharded
        models, accelerate's dispatch hooks move activations across shards
        during forward(); inputs only need to start on the device holding
        the first parameters."""
        return next(self.model.parameters()).device

    def format_prompt(self, template: str, sent: str, e1: str, e2: str) -> str:
        """Fill a prompt template. Overridden by EncoderMLMAdapter to also
        substitute a {mask} placeholder."""
        return template.format(sent=sent, e1=e1, e2=e2)

    def tokenize(self, text: str, max_len: int) -> tuple:
        """Tokenize a prompt.

        Args:
            text (str): prompt text.
            max_len (int): max nb of tokens.

        Returns:
            tuple: input_ids, attention_mask (both 1D tensors)
        """
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return encoded['input_ids'].view(-1), encoded['attention_mask'].view(-1)

    @abstractmethod
    def compute_target_index(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> int:
        """Index (into the sequence dim) of the hidden state to extract for
        this (unbatched) example."""

    @abstractmethod
    def compute_batch_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                                  target_index: torch.Tensor) -> torch.Tensor:
        """Run a batched forward pass and gather the per-row embedding at
        target_index. Returns a (batch, hidden) tensor on CPU."""


class EncoderMLMAdapter(BaseOreModel):
    """Adapter for masked-LM encoder models (BERT, RoBERTa, ModernBERT, ...)."""

    model_type = "encoder_mlm"

    def _load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.mask_id = self.tokenizer.mask_token_id
        if self.mask_id is None:
            raise ValueError(
                f"'{self.model_name}' has no mask token; it cannot be used "
                f"as an encoder_mlm model. Pass --model-type causal_lm if "
                f"this is actually a causal LM.")
        self.model = AutoModel.from_pretrained(
            self.model_name, device_map=self._device_map_arg())
        self.model.eval()

    def format_prompt(self, template, sent, e1, e2):
        if "{mask}" not in template:
            raise ValueError(
                f"Encoder-MLM prompt template must contain a '{{mask}}' "
                f"placeholder, got: {template!r}")
        return template.format(sent=sent, e1=e1, e2=e2, mask=self.tokenizer.mask_token)

    def compute_target_index(self, input_ids, attention_mask):
        matches = (input_ids == self.mask_id).nonzero().flatten()
        if matches.numel() != 1:
            raise ValueError(
                f"Expected exactly one mask token in the tokenized prompt, "
                f"found {matches.numel()} (mask_id={self.mask_id}). This "
                f"usually means max_len is too small and truncation removed "
                f"the mask token -- increase --max-len.")
        return matches.item()

    def compute_batch_embeddings(self, input_ids, attention_mask, target_index):
        input_ids = input_ids.to(self.input_device)
        attention_mask = attention_mask.to(self.input_device)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        arange = torch.arange(out.shape[0])
        embedding = out[arange, target_index.to(out.device)]
        return embedding.detach().float().to('cpu')


class CausalLMAdapter(BaseOreModel):
    """Adapter for causal (decoder-only) LMs (OLMo, Qwen, GPT-2, ...)."""

    model_type = "causal_lm"

    def __init__(self, model_name: str, device: str = None, quantization: str = None):
        self.quantization = quantization
        super().__init__(model_name, device=device)

    def _load(self):
        quant_config = None
        if self.quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.quantization == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        elif self.quantization not in (None, "none"):
            raise ValueError(
                f"Unknown quantization option: {self.quantization!r}; "
                f"expected null, '4bit', or '8bit'")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self._device_map_arg(),
            quantization_config=quant_config,
            dtype=torch.bfloat16 if quant_config is None else None,
        )
        self.model.eval()

    def compute_target_index(self, input_ids, attention_mask):
        # Computed from attention_mask directly so it is correct regardless
        # of padding_side, rather than assuming a fixed [:, -1, :] slice.
        real_positions = attention_mask.nonzero().flatten()
        return real_positions[-1].item()

    def compute_batch_embeddings(self, input_ids, attention_mask, target_index):
        input_ids = input_ids.to(self.input_device)
        attention_mask = attention_mask.to(self.input_device)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                              output_hidden_states=True)
            last_hidden = out.hidden_states[-1]
        arange = torch.arange(last_hidden.shape[0])
        embedding = last_hidden[arange, target_index.to(last_hidden.device)]
        return embedding.detach().float().to('cpu')


def create_ore_model(model_name: str, model_type: str = None,
                      device: str = None, quantization: str = None) -> BaseOreModel:
    """Factory: resolve the model family and instantiate the right adapter.

    Args:
        model_name (str): HF model name or local path.
        model_type (str, optional): override auto-detection ('encoder_mlm'/'causal_lm').
        device (str, optional): force a specific device; defaults to accelerate auto-sharding.
        quantization (str, optional): 'none'/'4bit'/'8bit'; only valid for causal_lm models.

    Returns:
        BaseOreModel: the loaded adapter.
    """
    resolved = detect_model_type(model_name, model_type)
    if resolved == "causal_lm":
        return CausalLMAdapter(model_name, device=device, quantization=quantization)

    if quantization not in (None, "none"):
        raise ValueError(
            f"--quantization is only supported for causal-LM models, but "
            f"'{model_name}' resolved to model_type='{resolved}'")
    return EncoderMLMAdapter(model_name, device=device)
