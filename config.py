# config.py
# Basic configuration and environment setup kept intentionally simple.

import warnings
from dotenv import load_dotenv

# Silence library warnings to avoid noisy logs.
warnings.filterwarnings("ignore")

# Load environment variables from a .env file if present.
load_dotenv()

# Absolute CSV paths as in the original code.
TRAIN_CSV = "/home/lavesh/medical-note-extraction/train.csv"
TEST_CSV = "/home/lavesh/medical-note-extraction/test.csv"
