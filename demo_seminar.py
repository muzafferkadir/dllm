"""
Seminar Demo: BD3LM Diffusion Language Model
=============================================
Qwen3-0.6B + BD3LM (Block Diffusion) + LoRA SFT (Alpaca)

Usage:
    PYTHONUTF8=1 .venv/Scripts/python.exe -u demo_seminar.py

Diffusion sürecini terminalde canlı olarak gösterir:
  - [MASK] tokenları adım adım çözülür
  - Progress bar ile ilerleme takip edilir
"""

import dllm
from peft import PeftModel

# ── Model yükleme ──
print("Model yükleniyor...")
class Args:
    model_name_or_path = ".models/a2d/Qwen3-0.6B"

model = dllm.utils.get_model(model_args=Args()).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=Args())

# LoRA adapter yükleme ve birleştirme
model = PeftModel.from_pretrained(
    model, ".models/a2d/Qwen3-0.6B/bd3lm/alpaca-sft-lora/checkpoint-final"
)
model = model.merge_and_unload()
print("Model hazır!\n")

# ── Sampler ayarları ──
sampler_config = dllm.core.samplers.BD3LMSamplerConfig(
    steps=128,
    max_new_tokens=128,
    block_size=32,
    temperature=0.2,
    remasking="low_confidence",
)
sampler = dllm.core.samplers.BD3LMSampler(model=model, tokenizer=tokenizer)
visualizer = dllm.utils.TerminalVisualizer(tokenizer=tokenizer)

# ── Demo promptları ──
demos = [
    "What is diffusion language modeling? Explain briefly.",
    "Write a short Python function to check if a number is prime.",
    "Explain the difference between autoregressive and diffusion models in NLP.",
]

for i, prompt in enumerate(demos):
    print(f"\n{'='*80}")
    print(f"  Demo {i+1}: {prompt}")
    print(f"{'='*80}\n")

    messages = [[{"role": "user", "content": prompt}]]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    outputs = sampler.sample(inputs, sampler_config, return_dict=True)

    # Diffusion sürecini görselleştir
    visualizer.visualize(outputs.histories, rich=True, fps=12, title=f"dLLM Demo {i+1}")

    # Son çıktıyı yazdır
    sequences = dllm.utils.sample_trim(tokenizer, outputs.sequences.tolist(), inputs)
    print(f"\n{'─'*80}")
    print(f"Final Output:")
    print(f"{'─'*80}")
    print(sequences[0].strip() if sequences[0].strip() else "<empty>")
    print()

    if i < len(demos) - 1:
        try:
            input("Enter'a basın sonraki demo için...")
        except EOFError:
            pass

print("\n✨ Demo tamamlandı!\n")
