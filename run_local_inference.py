# run.py
# Simple runner script: reads test.csv, invokes the chain row-by-row, saves outputs.

import time
import pandas as pd
from tqdm.auto import tqdm

from config import TEST_CSV
from schema_and_prompt import EXAMPLES_TEXT, parser
from model_chain import chain

def main():
    start_time = time.time()

    # Read test data.
    test = pd.read_csv(TEST_CSV)

    # Optional slicing as in the original script (uncomment one block at a time).
    # test = test[:900]        # ---- part 1
    # test = test[900:1800]    # ---- part 2
    # test = test[1800:2700]   # ---- part 3
    # test = test[2700:]       # ---- part 4

    # Iterate rows and run inference.
    for idx, row in tqdm(test.iterrows(), total=len(test)):
        result = chain.invoke(
            {
                "Note": row["Note"],
                "format_instructions": parser.get_format_instructions(),
                "EXAMPLES_TEXT": EXAMPLES_TEXT,
            }
        )
        # result[0] is full model output; result[1] is extracted JSON-only portion.
        test.at[idx, "json"] = result[1]
        test.at[idx, "full_response"] = result[0]
        print(f"Row {idx} Completed.")

    # Save outputs.
    test.to_csv("final_output_fewshot.csv", index=False)

    elapsed = time.time() - start_time
    print(f"Time elapsed: {elapsed:.1f}s")

if __name__ == "__main__":
    main()
