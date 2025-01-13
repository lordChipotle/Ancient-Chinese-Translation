## Ancient Chinese Translator + Phonology Fine-Tuning

This repository contains code and data for fine-tuning Google’s Gemma model to:

	1.	Translate Ancient Chinese texts into modern Chinese.
 
	2.	Provide historical pronunciations of characters or entire sentences, spanning multiple eras.

Overview

Ancient Chinese literature is rich and diverse, yet translating it accurately into modern Chinese and reproducing its historically accurate pronunciations can be challenging. Our project aims to tackle both tasks:

	1.	Translation: Convert classical/ancient Chinese text into understandable modern Chinese.
 
	2.	Phonology: Provide phonetic reconstructions for characters or sentences, referencing eras such as Middle Tang, Song, Yuan, Ming/Qing, or modern Mandarin.

By combining these features into a single model, researchers and enthusiasts can not only read ancient Chinese texts in contemporary language but also appreciate how characters were pronounced across different periods in China’s history.

Motivation and Challenge

	•	Translation Accuracy: Ancient Chinese has evolved significantly over centuries, and direct translation requires contextual knowledge of classical grammar and vocabulary.
	
 	•	Historical Pronunciation: Many modern learners have minimal exposure to the systematic phonological changes that occurred from the Tang Dynasty through the modern era.
  
	•	Integrated Approach: Typically, translation models ignore phonological features, while phonology-based methods rarely provide semantic translations.

Our challenge is to create an integrated pipeline that addresses both aspects simultaneously. By training a large language model with carefully curated data:

	•	Scholars can gain insights into philological nuances.
 
	•	Students of linguistics can learn how pronunciations have shifted over time.
 
	•	General users can more easily explore ancient texts without sacrificing historical context.

Data Sources

	1.	Erya Dataset (for Ancient-to-Modern Chinese Translation)
 
	•	We leveraged RUCAIBox/Erya-dataset to fine-tune Gemma on translating classical Chinese passages into modern Chinese.
 
	•	This parallel corpus covers historical texts from various dynasties, providing clean ancient→modern sentence pairs.
 
	2.	Ancient Chinese Phonology (ACP)
 
	•	Sourced from KaguraRuri/Ancient-Chinese-Phonology.
 
	•	This dataset maps thousands of Chinese characters to their reconstructed pronunciations across multiple historical periods.
 
	•	We integrate these phonological features into the same Gemma model, enabling it to output era-based phonetic reconstructions.
 

How This Addresses the Challenge

	1.	Multi-task Fine-Tuning
 
	•	By performing a two-phase LoRA fine-tuning, we first ensure Gemma can produce accurate translations. Then we incorporate phonological data to teach it how to generate character-level or sentence-level pronunciations across eras.
 
	•	The model learns to respond to user prompts that ask for both translation and historical pronunciations.
 
	2.	Era Identification (Optional)
 
	•	Our multi-task prompts can let the model “guess” or propose an era for a given text. Although the automatic era detection may not always be perfect, it provides a convenient starting point for phonological output.
 
	3.	User-Controlled Output
 
	•	Because we train Gemma using <start_of_turn> and <end_of_turn> tokens to delineate user instructions and model replies, we can craft precise prompts. For example:

## Repository Structure
You can either use our repo and download source data from original repo:

1. https://huggingface.co/datasets/RUCAIBox/Erya-dataset


2. https://github.com/KaguraRuri/Ancient-Chinese-Phonology/tree/main

or use the notebooks we created on Kaggle:
Finetune translation: https://www.kaggle.com/code/jesuisdanielhj/translator-of-ancient-chinese-literature
Finetune pronounciation: https://www.kaggle.com/code/jesuisdanielhj/multi-task-translator-phonology-fine-tuning-for

├── README.md  
├── requirements.txt  
├── src  
│   ├── config.py  
│   ├── model.py  
│   ├── preprocessing.py  
│   ├── train_translator.py  
│   └── finetune_phonology.py  
└── data (your input data, not tracked in Git)

1) Install dependencies from requirements.txt  
2) Prepare your data inside data/  
3) Adjust config.py to the correct model IDs, ranks, epochs, and file paths  
4) Run train_translator.py to train a translator  
5) Run finetune_phonology.py for the second-phase multi-task translator + phonology

## To use the model we finetuned:
Model weights: https://huggingface.co/lordChipotle/SimaQian

Usage:
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("lordChipotle/SimaQian")

model = AutoModelForCausalLM.from_pretrained("lordChipotle/SimaQian")

prompt = """ user Given the ancient text: 「子曰：學而時習之，不亦說乎？」

Identify the era
Provide the phonetic reading
Translate into modern Chinese model """
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, max_length=256)

print(tokenizer.decode(outputs[0]))
