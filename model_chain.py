# model_chain.py
# Model loading, quantization config, and runnable chain assembly.

import torch
from transformers import BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from schema_and_prompt import prompt, parser, EXAMPLES_TEXT

# 4-bit quantization config for efficient inference.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# HF pipeline with the specified model and generation parameters.
pipeline = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-14B-Instruct",
    task="text-generation",
    model_kwargs={"quantization_config": bnb_config, "device_map": "auto"},
    pipeline_kwargs={"max_new_tokens": 1000, "temperature": 0.1},
)

# Extract only the final JSON after the last "Assistant:" token occurrence.
def AssistantReponseExtractor(text: str) -> str:
    return text.split("Assistant:")[6].strip()

# Two parallel branches: raw text and post-processed JSON-only branch.
parallel_chain = RunnableParallel(
    {
        "without_parser": RunnablePassthrough(),
        "with_parser": RunnableLambda(AssistantReponseExtractor),
    }
)

# Combine both branches into a list: [full_response, json_only]
def combine_both(results):
    return [results["without_parser"], results["with_parser"]]

# Final chain: prompt -> model -> parallel split -> combine.
chain = prompt | pipeline | parallel_chain | RunnableLambda(combine_both)
