# Medical Note Extraction - H2O.ai Competition (1st Place Solution)

This repository contains the 1st place solution for the [H2O.ai Competition: Medical Note Extraction](https://www.kaggle.com/competitions/medical-note-extraction-h-2-o-gen-ai-world-ny). The goal was to transform unstructured, synthetically generated medical notes into a structured JSON format.

This solution achieved a score of **0.99031** on the private leaderboard.

## Competition Overview

Unstructured medical notes are a major challenge in healthcare, making systematic analysis difficult and error-prone. This competition aimed to solve this by developing methods to convert free-text notes into standardized, structured JSON. This enables advanced analytics, predictive modeling, and AI-driven insights to improve patient outcomes.

The evaluation was based on a custom metric that checked for valid JSON parsing and then calculated similarity for different data types (numerical, string, list/set, dictionaries) against the ground truth.

## Our Approach

The core of this solution is an LLM-based extraction pipeline built with LangChain, using a powerful open-source model with 4-bit quantization and a meticulously crafted prompt.

The high-level pipeline is:
**`Medical Note`** -> **`LangChain ChatPromptTemplate`** (with Few-Shot Examples) -> **`LLM (Qwen2.5-14B-Instruct)`** -> **`Custom Parser`** -> **`Post-processing (Normalization)`** -> **`Structured JSON`**

### 1. LLM and Model Chain

* **Model:** `Qwen/Qwen2.5-14B-Instruct` was used as the base model.
* **Quantization:** The model was loaded in 4-bit using `transformers.BitsAndBytesConfig` (nf4, float16 compute) for efficient inference on a single GPU (A100 80GB).
* **Pipeline:** The model was wrapped in a `langchain_huggingface.HuggingFacePipeline` set to the `"text-generation"` task.

### 2. Schema and Prompt Engineering

This was the most critical part of the solution.

* **Pydantic Schema:** A detailed Pydantic `BaseModel` (`JsonOutput`) was defined in `schema_and_prompt.py` to specify the exact output structure.
    * `Annotated`, `Literal`, and `Field` types were used to provide precise descriptions and constraints for each field (e.g., patient age, gender, vital signs).
    * The allowed values for `visit_motivation` and `symptoms` were dynamically loaded from the `train.csv` file to create strict `Literal` types, ensuring the model only chose from valid options.
* **Prompt Template:** A `ChatPromptTemplate` was engineered with:
    1.  A **System Prompt** instructing the model to act as a precise medical extractor and follow strict JSON-only output rules.
    2.  The `format_instructions` from a `PydanticOutputParser` instance, telling the model the exact schema.
    3.  A **Human Prompt** that included high-quality **few-shot examples** (`EXAMPLES_TEXT`) showing the expected `Medical Note` -> `Assistant: {JSON}` format.
    4.  The actual `{Note}` to be processed.

### 3. Custom Parsing and Chain Execution

* **Challenge:** The standard `PydanticOutputParser` was not robust enough to handle the model's raw output directly, which sometimes contained repeated "Assistant:" tokens.
* **Solution:** A custom parsing function (`AssistantReponseExtractor`) was created to split the raw text and reliably extract the final JSON block.
* **LangChain Runnable:** The final chain used `RunnableParallel` to process the model output in two ways simultaneously: one raw (`without_parser`) and one passed through the custom JSON extractor (`with_parser`). This allowed for saving both the full model response (for debugging) and the cleaned JSON.

### 4. Post-processing and Submission

The `submission_builder.py` script handles the final steps to create a competition-ready submission:

1.  **Combine Parts:** The test set inference was run in 4 parallel chunks due to the long processing time (avg. 40 secs/note). This script combines the partial output CSVs.
2.  **Clean Nulls:** A recursive function (`remove_nulls`) cleans the extracted JSON by removing any keys with `None` values, as these were not required by the schema.
3.  **Normalize Symptoms:** A mapping dictionary (`sym_mapping`) is used to normalize extracted symptoms (e.g., mapping "persistent_cough" to "cough") and deduplicate the list.
4.  **Normalize Visit Motivation:** A similar mapping (`vm_mapping`) ensures `visit_motivation` values are standardized (e.g., mapping "Asthma (Exacerbation)" to "Asthma").
5.  **Final CSV:** The script outputs the final `submission_llm.csv` with just the `ID` and cleaned `json` columns.

## Challenges Faced

1.  **Pydantic Parser Failure:** The standard parser couldn't handle imperfections in the LLM's output. This was solved by creating a custom `RunnableLambda` extractor to manually parse the JSON string from the model's text response.
2.  **Repeated Model Output:** Some notes caused the model to repeat the "Assistant:" token multiple times. The custom extractor was made robust to this by splitting on the token and taking the final valid block.
3.  **Inference Time:** With ~3700 test rows at 40s/row, full inference was too long. The solution was to split the `test.csv` into four parts, run four instances of the script in parallel on the same A100 GPU, and then combine the results.

## How to Run

### 1. Setup

1.  Clone this repository.
2.  Create and activate a Python virtual environment.
3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
   

### 2. Configuration

1.  Download the competition data (`train.csv`, `test.csv`).
2.  Update the absolute paths for `TRAIN_CSV` and `TEST_CSV` in `config.py` to point to your local data files.
    *Note: `train.csv` is required at runtime to build the list of valid symptoms and visit motivations for the prompt schema*.

### 3. Run Inference

1.  To run inference on the full test set, execute:
    ```bash
    python run_local_inference.py
    ```
   
2.  This will create a `final_output_fewshot.csv` file containing the note, the full model response, and the extracted JSON.
    *Note: As noted in "Challenges", you may want to modify `run_local_inference.py` to run on slices of the test set if you face time or memory constraints.*

### 4. Build Submission

1.  After running inference (and naming your output files e.g., `final_output_fewshot_part1.csv`, etc., if you ran in chunks), update `submission_builder.py` to load your specific output files.
2.  Run the submission builder:
    ```bash
    python submission_builder.py
    ```
3.  This will generate the final `submission_llm.csv` file, ready for upload.

## Key Files in This Repository

* `config.py`: Holds file paths and basic configuration.
* `schema_and_prompt.py`: Defines the core `Pydantic` output schema and the `ChatPromptTemplate` (including few-shot examples).
* `model_chain.py`: Configures the `Qwen2.5` model, 4-bit quantization, and assembles the final `LangChain` runnable chain with the custom parser.
* `run_local_inference.py`: The main script to iterate through `test.csv`, invoke the chain, and save results.
* `submission_builder.py`: Post-processing script to combine results, clean nulls, and normalize symptoms/visit motivations for the final submission.
* `requirements.txt`: A list of all necessary Python packages.

## Tools and Frameworks

* **LangChain:** For the core runnable chain, prompt templates, and model integration.
* **Pydantic:** For robust schema definition and validation.
* **HuggingFace `transformers`:** For the `HuggingFacePipeline`.
* **`bitsandbytes`:** For 4-bit model quantization.

## **Team Members**  
This solution is a result of our collaborative efforts:  
- **Lavesh Kadam**  
- **Mannan Thakur**  
- **Rushikesh Khandetod**  
- **Archisman Bera**  
