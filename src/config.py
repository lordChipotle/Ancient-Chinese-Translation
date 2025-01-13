"""
Configuration file for hyperparameters, file paths, etc.
"""

import os

# Kaggle data (Gemma)
GEMMA_LANGUAGE_TUNING_PATH = "gemma-language-tuning"
GEMMA_INSTRUCT_MODEL_PATH = "keras/gemma2/Keras/gemma2_instruct_2b_en/1"
GEMMA_MODEL_ID = "gemma2_instruct_2b_en"  # Model preset name, e.g. "gemma2_instruct_2b_en"

# Environment settings (Optional)
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

# LoRA configurations
LORA_RANK = 4
LORA_RANK_2 = 4  # second-phase rank if different
TRAIN_EPOCHS = 3
TRAIN_EPOCHS_2 = 2
LEARNING_RATE = 1e-4
LEARNING_RATE_2 = 1e-5
TOKEN_LIMIT = 256

# Where to save or load LoRA weights
FIRST_LORA_SAVE_PREFIX = "/kaggle/working/translator_chinese"
SECOND_LORA_SAVE_PREFIX = "/kaggle/working/translator_plus_phonology"

# Example data paths
DATA_FOLDER = "/kaggle/working/dataset"
ACP_CSV_PATH = f"{DATA_FOLDER}/ancient_chinese_phonology.csv"
ERYA_SUBFOLDERS = ["shij", "mings", "hans", "xint", "xux"]  # example Erya subfolders 