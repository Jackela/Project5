"""
Defines project-wide constants, primarily for path management.
Establishes a Single Source of Truth (SSOT) for all directory lookups.
"""

from pathlib import Path

# --- Corrected Project Root Definition ---
# This is the single source of truth for the project's root directory.
# It robustly finds the root by looking for a known file ('README.md').
# All other paths in the application should be derived from this constant.
def find_project_root() -> Path:
    """Dynamically finds the project's root directory."""
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if (parent / 'README.md').exists() and (parent / 'p5_text_classification').is_dir():
            return parent
    raise FileNotFoundError("Could not find project root. Make sure 'README.md' and 'p5_text_classification' exist at the root.")

try:
    PROJECT_ROOT = find_project_root()
except FileNotFoundError as e:
    # Provide a fallback for scenarios where the script is run in an unusual context
    print(f"Warning: Could not dynamically find project root. Falling back to relative pathing. Error: {e}")
    PROJECT_ROOT = Path('.').resolve().parent.parent # Fallback

# --- Standard Directory Layout ---
# All paths are derived from the single source of truth, PROJECT_ROOT.
CONFIG_DIR = PROJECT_ROOT / "p5_text_classification" / "configs"
DATA_DIR = PROJECT_ROOT / "p5_text_classification" / "data"

# The REPORTS_DIR must be at the top level of the project.
REPORTS_DIR = PROJECT_ROOT / "REPORTS"

# Subdirectories for organized output
LOGS_DIR = REPORTS_DIR / "logs"
PLOTS_DIR = REPORTS_DIR / "plots"
RESULTS_DIR = REPORTS_DIR / "results"
TRAINING_HISTORY_DIR = REPORTS_DIR / "training_history"

# --- Create Directories ---
# Ensure all output directories exist on module import.
# This prevents FileNotFoundError during runtime.
REPORTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
(PLOTS_DIR / "confusion_matrices").mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
TRAINING_HISTORY_DIR.mkdir(exist_ok=True)


"""
Constants module implementing Single Source of Truth (SSOT) principle.
All constants are defined here to avoid duplication and hardcoding.
"""

# Character alphabet as specified in the VDCNN paper
# This is the single source of truth for character encoding
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+=<>()[]{} "

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# Token IDs (these should match the configuration)
PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1

# Derived constants
VOCAB_SIZE = len(ALPHABET) + 2  # +2 for PAD and UNK tokens
ALPHABET_SIZE = len(ALPHABET)

# Model defaults
DEFAULT_SEQUENCE_LENGTH = 1014
DEFAULT_EMBEDDING_DIM = 16

# Training defaults
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_EPOCHS = 10

# File extensions
YAML_EXTENSION = ".yaml"
MODEL_EXTENSION = ".pth"
LOG_EXTENSION = ".log" 