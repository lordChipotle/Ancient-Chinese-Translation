"""
Stage 1: Translator fine-tuning using Erya data (ancient -> modern).
Equivalent to translator_of_ancient_chinese_literature.py
"""

import os
import time
import matplotlib.pyplot as plt
from config import (
    TOKEN_LIMIT, LORA_RANK, TRAIN_EPOCHS, LEARNING_RATE,
    FIRST_LORA_SAVE_PREFIX
)
from model import load_gemma_model, enable_lora, compile_gemma
from preprocessing import (
    build_erya_datasets, build_translation_prompts, text_gen_example
)
import keras

def train_translator():
    # 1) Load model & tokenizer
    gemma, tokenizer = load_gemma_model()

    # 2) Prepare data
    erya_data = build_erya_datasets()
    train_prompts = build_translation_prompts(erya_data["train"], tokenizer)
    valid_prompts = build_translation_prompts(erya_data["validation"], tokenizer)

    print("Train samples:", len(train_prompts))
    print("Valid samples:", len(valid_prompts))
    print("Example sample:\n", train_prompts[0])

    # 3) LoRA
    enable_lora(gemma, LORA_RANK)
    compile_gemma(gemma, LEARNING_RATE)

    # 4) Callback
    class SaveLoraCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            lora_path = f"{FIRST_LORA_SAVE_PREFIX}_{LORA_RANK}_epoch{epoch+1}.lora.h5"
            gemma.backbone.save_lora_weights(lora_path)
            print(f"Saved LoRA weights to: {lora_path}")
            text_gen_example(gemma, "子曰：學而時習之，不亦說乎？")
            text_gen_example(gemma, "昔在黃帝時，天下太平，萬民和樂。")

    # 5) Fit
    history = gemma.fit(
        train_prompts,
        epochs=TRAIN_EPOCHS,
        batch_size=1,
        validation_data=(valid_prompts,),
        callbacks=[SaveLoraCallback()],
    )

    # 6) Plot
    if hasattr(history, "history"):
        plt.plot(history.history["loss"], label="train_loss")
        if "val_loss" in history.history:
            plt.plot(history.history["val_loss"], label="val_loss")
        plt.title("Translator Fine-Tuning Loss")
        plt.legend()
        plt.show()

    # 7) Test
    text_gen_example(gemma, "太史公曰：匈奴之盛自冒頓始。")

if __name__ == "__main__":
    train_translator() 