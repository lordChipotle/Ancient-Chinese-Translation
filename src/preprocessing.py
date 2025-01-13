"""
Data loading and prompt-building utilities.
"""

import os
import pandas as pd
import time
from datasets import Dataset, DatasetDict

from config import DATA_FOLDER, ACP_CSV_PATH, ERYA_SUBFOLDERS, TOKEN_LIMIT

def load_ancient_chinese_phonology():
    """
    Reads the ancient Chinese phonology CSV and returns a DataFrame.
    The CSV is expected to contain columns like: character, MiddleTang, LateTang, etc.
    """
    df = pd.read_csv(ACP_CSV_PATH)
    return df

def build_single_char_phonology_prompts(df, tokenizer):
    """Train the model to output phonology for single characters (no era guess)."""
    prompts = []
    for _, row in df.iterrows():
        char = str(row["character"])
        middle_tang = str(row.get("MiddleTang", "") or "")
        late_tang   = str(row.get("LateTang", "") or "")
        song        = str(row.get("Song", "") or "")
        yuan        = str(row.get("Yuan", "") or "")
        ming_qing   = str(row.get("MingQing", "") or "")
        mandarin    = str(row.get("Mandarin", "") or "")

        user_prompt = (
            f"<start_of_turn>user\n"
            f"Character: {char}\n"
            "Please provide the historical pronunciations.\n"
            f"<end_of_turn>\n"
        )
        model_response = (
            f"<start_of_turn>model\n"
            f"MiddleTang: {middle_tang}\n"
            f"LateTang: {late_tang}\n"
            f"Song: {song}\n"
            f"Yuan: {yuan}\n"
            f"MingQing: {ming_qing}\n"
            f"Mandarin: {mandarin}\n"
            f"<end_of_turn>"
        )
        text = user_prompt + model_response
        length = len(tokenizer(text))
        if length < TOKEN_LIMIT:
            prompts.append(text)
    return prompts

def build_multi_task_prompts(data_list, tokenizer):
    """Teach the model to respond with (1) era, (2) reading, (3) translation for sentences."""
    prompts = []
    for item in data_list:
        ancient_text = item["text"]
        era_label = item["era"]
        reading = item["reading"]
        modern = item["modern"]

        user_prompt = (
            f"<start_of_turn>user\n"
            f"Given the ancient Chinese sentence: 「{ancient_text}」\n"
            "1) Identify the historical era\n"
            "2) Provide the sentence-level pronunciation\n"
            "3) Provide the modern Chinese translation\n"
            f"<end_of_turn>\n"
        )
        model_response = (
            f"<start_of_turn>model\n"
            f"Era: {era_label}\n"
            f"Pronunciation: {reading}\n"
            f"Translation: {modern}\n"
            f"<end_of_turn>"
        )
        text = user_prompt + model_response

        length = len(tokenizer(text))
        if length < TOKEN_LIMIT:
            prompts.append(text)
    return prompts

def build_erya_datasets():
    """
    Load Erya data from subfolders: train.src, train.tgt, etc.
    Returns a HuggingFace DatasetDict with 'train', 'validation', 'test'.
    """
    def load_parallel(src_file, tgt_file):
        anc, mod = [], []
        with open(src_file, "r", encoding="utf-8") as fsrc, \
             open(tgt_file, "r", encoding="utf-8") as ftgt:
            for a_line, m_line in zip(fsrc, ftgt):
                anc.append(a_line.strip())
                mod.append(m_line.strip())
        return anc, mod

    base_path = DATA_FOLDER
    from datasets import Dataset, DatasetDict

    train_anc_all, train_mod_all = [], []
    valid_anc_all, valid_mod_all = [], []
    test_anc_all,  test_mod_all  = [], []

    for sub in ERYA_SUBFOLDERS:
        folder_path = f"{base_path}/{sub}"
        train_src_path = f"{folder_path}/train.src"
        train_tgt_path = f"{folder_path}/train.tgt"
        valid_src_path = f"{folder_path}/valid.src"
        valid_tgt_path = f"{folder_path}/valid.tgt"
        test_src_path  = f"{folder_path}/test.src"
        test_tgt_path  = f"{folder_path}/test.tgt"

        # accumulate
        if os.path.exists(train_src_path) and os.path.exists(train_tgt_path):
            anc_lines, mod_lines = load_parallel(train_src_path, train_tgt_path)
            train_anc_all.extend(anc_lines)
            train_mod_all.extend(mod_lines)

        if os.path.exists(valid_src_path) and os.path.exists(valid_tgt_path):
            anc_lines, mod_lines = load_parallel(valid_src_path, valid_tgt_path)
            valid_anc_all.extend(anc_lines)
            valid_mod_all.extend(mod_lines)

        if os.path.exists(test_src_path) and os.path.exists(test_tgt_path):
            anc_lines, mod_lines = load_parallel(test_src_path, test_tgt_path)
            test_anc_all.extend(anc_lines)
            test_mod_all.extend(mod_lines)

    train_dataset = Dataset.from_dict({"ancient": train_anc_all, "modern": train_mod_all})
    valid_dataset = Dataset.from_dict({"ancient": valid_anc_all, "modern": valid_mod_all})
    test_dataset  = Dataset.from_dict({"ancient": test_anc_all,  "modern": test_mod_all})

    return DatasetDict({
        "train": train_dataset,
        "validation": valid_dataset,
        "test": test_dataset
    })

def build_translation_prompts(ds, tokenizer):
    prompt_data = []
    for ancient, modern in zip(ds["ancient"], ds["modern"]):
        text = (
            f"<start_of_turn>user\n{ancient}<end_of_turn>\n"
            f"<start_of_turn>model\n{modern}<end_of_turn>"
        )
        length = len(tokenizer(text))
        if length < TOKEN_LIMIT:
            prompt_data.append(text)
    return prompt_data

def text_gen_example(model, prompt):
    """
    Quick generation test.
    """
    input_text = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    output = model.generate(input_text, max_length=TOKEN_LIMIT)
    print("\nGemma output:")
    print(output) 