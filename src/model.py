"""
Utility to set up the Gemma model and LoRA.
"""

import keras
import keras_nlp
from config import GEMMA_MODEL_ID, TOKEN_LIMIT

def load_gemma_model():
    """Load the base Gemma model and tokenizer."""
    tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(GEMMA_MODEL_ID)
    gemma = keras_nlp.models.GemmaCausalLM.from_preset(GEMMA_MODEL_ID)
    gemma.preprocessor.sequence_length = TOKEN_LIMIT
    return gemma, tokenizer

def enable_lora(model, rank):
    """Enable LoRA for the Gemma backbone."""
    model.backbone.enable_lora(rank=rank)
    return model

def compile_gemma(model, lr_value=1e-4):
    optimizer = keras.optimizers.AdamW(learning_rate=lr_value, weight_decay=0.01)
    # exclude certain parameter names from weight decay if needed
    optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model 