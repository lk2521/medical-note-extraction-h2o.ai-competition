# README.md

# Medical Note Extraction to JSON
Transforms brief medical notes into a strict JSON schema for competition submission, emphasizing parseability and type-correct outputs for reliable scoring. [web:25]

## Overview
Unstructured clinical notes are standardized into structured JSON to enable analytics and evaluation with the competition’s custom JSON-based scorer. [web:24]

## Data and Output
- Input: CSV with a Note column per row containing the clinical note text to be extracted. [web:24]
- Output: Submission CSV with columns [ID, json], where json is a single serialized JSON object per row. [file:21]

## Evaluation (summary)
The scorer first checks that json parses, then applies type-aware similarity across nested structures, so valid, type-correct, and schema-consistent JSON is essential. [web:24]

## Method
- Minimal text cleanup, then a strict prompt with few-shot examples and format instructions to elicit a single JSON object. [web:24]
- Typed schema via Pydantic models with explicit field types and enumerations aligned to train-derived vocabularies. [web:24]
- Post-processing validates JSON, removes nulls recursively, and normalizes values before writing the final submission. [file:21]

## Project structure
medical_note_extraction/
├─ config.py # Paths and environment setup
├─ schema_and_prompt.py # Pydantic schema, parser, examples, prompt
├─ model_chain.py # HF pipeline + LangChain runnables
├─ run.py # Inference over test.csv to part/full outputs
├─ submission_builder.py # Consolidation, validation, cleaning, submission
└─ requirements.txt # Dependencies

[web:25]

## Setup
- Create a virtual environment and install dependencies: pip install -r requirements.txt. [web:25]
- Set train/test CSV paths in config.py or align files to the defaults used by the scripts. [web:25]

## Run
- Extraction: python run.py to generate per-part or full outputs with json and full_response columns. [web:24]
- Build submission: python submission_builder.py to validate/clean and write /kaggle/working/submission.csv as [ID, json]. [file:21]

## Post-processing details
- Consolidates part files, verifies json is a dict, and attempts recovery from full_response when needed. [file:21]
- Removes nulls recursively and applies mapping-based normalization for controlled vocabularies. [file:21]

## Notes and tips
- Some generations may repeat “Assistant:” blocks; recovery uses a consistent split index to isolate the final JSON segment. [file:21]
- Splitting test rows into parts and merging later improves wall-time without changing extraction logic. [file:21]
