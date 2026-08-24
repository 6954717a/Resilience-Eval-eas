# Language-model backends

The classes in this directory are instantiated by Hydra-backed planners and
evaluation runners. They are not standalone services.

## Configuration

Model configuration lives in `habitat_llm/conf/llm/`:

- `llama.yaml` and `llama_non_instruct.yaml` configure local or Hugging Face
  Llama-compatible models.
- `qwen.yaml` reads `QWEN_MODEL_PATH`.
- `qwen2.5.yaml` reads `QWEN25_MODEL_PATH`.
- The Qwen 3.5 vLLM baseline reads `QWEN35_SERVED_MODEL`.
- `openai_chat.yaml` configures the Azure OpenAI chat adapter.
- `multimodal_llama.yaml` configures image-and-text generation.

Set model paths through environment variables or Hydra overrides. Do not commit
machine-local paths, access keys, or deployment URLs.

The Azure adapter requires:

```text
OPENAI_API_KEY
OPENAI_ENDPOINT
```

OpenAI-compatible critic and judge configurations read credentials from
`OPENAI_API_KEY`. Set `HABITAT_LLM_BASE_URL` to replace their loopback default;
the perception connector also accepts the legacy `LLM_BASE_URL` name. The
optional text encoder can be changed with `TEXT_ENCODER_MODEL`.

## Programmatic use

```python
from habitat_llm.llm import instantiate_llm

model = instantiate_llm(
    "llama_non_instruct",
    generation_params={"engine": "path-or-hugging-face-model-id"},
)
response = model.generate("The answer is", max_length=10)
```

Multimodal prompts are sequences of `("text", value)` and `("image", data_url)`
items. Convert local images to data URLs with
`habitat_llm.llm.instruct.utils.pil_image_to_data_url` before generation.
