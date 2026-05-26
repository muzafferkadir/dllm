from dataclasses import dataclass

import torch
import transformers
import tyro

import dllm

A2D_CONFIG_MAP = {
    "llama": dllm.pipelines.a2d.A2DLlamaConfig,
    "qwen2": dllm.pipelines.a2d.A2DQwen2Config,
    "qwen3": dllm.pipelines.a2d.A2DQwen3Config,
    "qwen3_5_text": dllm.pipelines.a2d.A2DQwen3_5Config,
}


@dataclass
class ScriptArguments:
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B"
    output_dir: str = ".models/a2d/Qwen2.5-0.5B"
    random_init: bool = False

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


def main():

    args = tyro.cli(ScriptArguments)
    dllm.utils.print_args(args)

    # Load source config to detect model type
    src_config_dict = transformers.AutoConfig.from_pretrained(args.model_name_or_path).to_dict()
    is_multimodal = src_config_dict.get("model_type") in ("qwen3_5", "qwen3_5_moe")

    if is_multimodal:
        # Multimodal model: extract text backbone
        print(f"Detected multimodal model ({src_config_dict['model_type']}), extracting text backbone...")
        src_full_model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
        )
        # Get text model and its config
        src_text_model = src_full_model.model.language_model if hasattr(src_full_model.model, 'language_model') else src_full_model.model
        src_config = src_text_model.config
        # A2D LMHeadModel wraps the text model under "model.", so add prefix
        raw_sd = src_text_model.state_dict()
        src_state_dict = {f"model.{k}": v for k, v in raw_sd.items()}
        # Copy lm_head
        if hasattr(src_full_model, 'lm_head'):
            for k, v in src_full_model.lm_head.state_dict().items():
                src_state_dict[f"lm_head.{k}"] = v
        del src_full_model, src_text_model  # free memory
    else:
        # Standard causal LM
        src_model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            dtype="bfloat16",
        )
        src_config = src_model.config
        src_state_dict = src_model.state_dict()

    src_tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
    )

    # Remove unused HF fields
    for k in ["auto_map", "architectures"]:
        if hasattr(src_config, k):
            delattr(src_config, k)

    # Select corresponding A2D config class
    base_type = src_config.model_type
    tgt_config_cls = A2D_CONFIG_MAP[base_type]

    # Build A2D config from source config dict
    cfg_dict = src_config.to_dict()
    cfg_dict.pop("model_type", None)
    tgt_config = tgt_config_cls(**cfg_dict)

    with dllm.utils.init_device_context_manager():
        tgt_model = transformers.AutoModel.from_config(tgt_config)

        if not args.random_init:
            missing, unexpected = tgt_model.load_state_dict(
                src_state_dict, strict=False
            )
            print("missing:", missing)
            print("unexpected:", unexpected)

        # Save model and config
        tgt_model.save_pretrained(args.output_dir)
        tgt_config.save_pretrained(args.output_dir)
        src_tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
