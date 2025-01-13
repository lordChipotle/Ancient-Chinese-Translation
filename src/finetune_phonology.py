"""
Stage 2: Multi-Task Translator + Phonology Fine-Tuning
Equivalent to multi_task_translator_+_phonology_fine_tuning_for.py
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
from config import (
    TOKEN_LIMIT, LORA_RANK_2, TRAIN_EPOCHS_2, LEARNING_RATE_2,
    SECOND_LORA_SAVE_PREFIX, FIRST_LORA_SAVE_PREFIX
)
from model import load_gemma_model, enable_lora, compile_gemma
from preprocessing import (
    load_ancient_chinese_phonology,
    build_single_char_phonology_prompts,
    build_multi_task_prompts,
    text_gen_example
)
import keras

def finetune_phonology():
    # 1) Load model & apply the first-phase LoRA
    gemma, tokenizer = load_gemma_model()
    previous_lora_path = f"{FIRST_LORA_SAVE_PREFIX}_4_epoch3.lora.h5"  # Example
    if os.path.exists(previous_lora_path):
        gemma.backbone.load_lora_weights(previous_lora_path)
        print("Loaded previous translator LoRA weights successfully!")

    # 2) Build single-char phonology data
    df_acp = load_ancient_chinese_phonology()
    single_char_prompts = build_single_char_phonology_prompts(df_acp, tokenizer)

    # 3) Build multi-task data
    sample_anc_sents = [
        {
            "text": "子曰：學而時習之，不亦說乎？",
            "era": "Spring and Autumn",
            "reading": "tsəi i̯wət ... [fake example]",
            "modern": "Confucius said: To study and practice frequently, is that not a delight?"
        },
        {
            "text": "太史公曰：匈奴之盛自冒頓始。",
            "era": "Han",
            "reading": "tʰai ɕi kʰuŋ i̯wət ... [fake example]",
            "modern": "The Grand Historian said: The Xiongnu's prosperity began with Maodun."
        },
    ]
    multi_task_prompts = build_multi_task_prompts(sample_anc_sents, tokenizer)

    all_prompts = []
    all_prompts.extend(single_char_prompts)
    all_prompts.extend(multi_task_prompts)
    print("Total training samples (second-phase):", len(all_prompts))

    # 4) Enable LoRA again for second-phase
    enable_lora(gemma, LORA_RANK_2)
    compile_gemma(gemma, LEARNING_RATE_2)

    # 5) Custom callback
    class CustomCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # Save LoRA
            model_name = f"{SECOND_LORA_SAVE_PREFIX}_{LORA_RANK_2}_epoch{epoch+1}.lora.h5"
            gemma.backbone.save_lora_weights(model_name)
            print(f"Saved second-phase LoRA to: {model_name}")
            # Quick test:
            text_gen_example(gemma, "Character: 童\nPlease provide the historical pronunciations.")
            text_gen_example(
                gemma,
                "Given the ancient Chinese sentence: 「子曰：學而時習之，不亦說乎？」"
                "1) Identify the historical era\n"
                "2) Provide the sentence-level pronunciation\n"
                "3) Provide the modern Chinese translation"
            )

    # 6) Fine-Tune
    history = gemma.fit(
        all_prompts,
        epochs=TRAIN_EPOCHS_2,
        batch_size=1,
        validation_split=0.05,
        callbacks=[CustomCallback()],
    )
    if hasattr(history, "history"):
        plt.plot(history.history["loss"], label="train_loss")
        if "val_loss" in history.history:
            plt.plot(history.history["val_loss"], label="val_loss")
        plt.title("Second-Phase Fine-Tuning Loss")
        plt.legend()
        plt.show()

    # 7) Final usage example
    final_lora_path = f"{SECOND_LORA_SAVE_PREFIX}_{LORA_RANK_2}_epoch{TRAIN_EPOCHS_2}.lora.h5"
    gemma.backbone.load_lora_weights(final_lora_path)

    text_gen_example(gemma, "Character: 冻\nPlease provide the historical pronunciations.")
    text_gen_example(
        gemma,
        "Given the ancient Chinese sentence: 「太史公曰：匈奴之盛自冒頓始。」"
        "1) Identify the historical era\n"
        "2) Provide the sentence-level pronunciation\n"
        "3) Provide the modern Chinese translation"
    )

if __name__ == "__main__":
    finetune_phonology() 