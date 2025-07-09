# How to Run Experiments: A User Guide

## 1. Overview
This guide provides detailed instructions for using the automation scripts to run experiments, analyze results, and generate reports. The project is designed to be operated primarily through the scripts located in the `p5_text_classification/scripts/` directory. This script-driven approach ensures consistency, simplifies complex workflows, and automates result aggregation.

All commands should be run from the project's root directory.

## 2. The Standard Workflow
A typical workflow for this project involves three main steps, each corresponding to a specific script:

1.  **`run_quick_tests.py`**: Perform a quick, small-scale run of all model-dataset combinations. **This is the recommended first step** to verify your environment and ensure the entire code pipeline is functional.
2.  **`run_full_experiments.py`**: Launch the complete, pre-configured experiments. This is computationally intensive and will train the models on the full datasets to reproduce or surpass the paper's results.
3.  **`generate_report.py`**: After experiments are complete, run this script to parse all logs and prediction files from the `REPORTS/` directory, generate comparison plots, create confusion matrices, and compile everything into a final Markdown report.

---

## 3. Step-by-Step Instructions

### Step 1: Quick Functionality Validation
This script is designed as a "smoke test." It runs each model (VDCNN, DenseNet) on each dataset (Yelp, Yahoo) using a very small subset of data (`max_samples=1000`) for only a few epochs. It should complete in a few minutes.

**Command:**
```bash
python p5_text_classification/scripts/run_quick_tests.py
```
- **Expected Outcome**: The script will print progress to the console and confirm that each test run completes successfully. It will generate a summary `quick_test_report.md` in the `REPORTS/` directory. If this step fails, there is likely an issue with your environment setup or dependencies.

### Step 2: Running Full-Scale Experiments
This script is the main workhorse of the project. It reads the configurations in `p5_text_classification/configs/training/` and executes each one sequentially. This process can take many hours, especially on a machine without a powerful GPU.

**Command:**
```bash
python p5_text_classification/scripts/run_full_experiments.py
```
- **Expected Outcome**: For each experiment, you will see detailed training progress logged to the console. The script will automatically save the following to the `REPORTS/` directory:
    - **Logs**: A detailed `.log` file for each run in `REPORTS/logs/`.
    - **Best Model**: The `.pth` model file with the highest validation accuracy in `REPORTS/`.
    - **Predictions**: A `.csv` file containing the model's predictions on the test set in `REPORTS/`.

### Step 3: Generating the Final Analysis Report
After one or more full experiments have finished, this script aggregates all the results into a comprehensive analysis.

**Command:**
```bash
python p5_text_classification/scripts/generate_report.py
```
- **Expected Outcome**: The script will:
    1.  Parse all `.log` files in `REPORTS/logs/`.
    2.  Generate learning curve plots (loss/accuracy vs. epoch) for each experiment and save them to `REPORTS/plots/`.
    3.  Generate confusion matrices from the prediction `.csv` files and save them to `REPORTS/plots/confusion_matrices/`.
    4.  Create a summary report named `experiment_analysis.md` in the `REPORTS/` directory, which includes comparison tables and all the generated plots.

---

## 4. Advanced Usage: Overriding Parameters
While the `run_full_experiments.py` script uses the exact configurations from the YAML files, it also allows you to override key parameters from the command line for quick, targeted tests without modifying the config files.

| Argument | Type | Description |
|:---|:---|:---|
| `--epochs` | Integer | Override the number of training epochs for all experiments. |
| `--batch_size`| Integer | Override the batch size for all experiments. |
| `--lr` | Float | Override the learning rate for all experiments. |
| `--max_samples`| Integer | Limit the number of samples to use from each dataset. |

**Example:**
Run all experiments, but only for 5 epochs and with a maximum of 10,000 samples from each dataset. This is useful for a moderately fast run that still produces meaningful results.

```bash
python p5_text_classification/scripts/run_full_experiments.py --epochs 5 --max_samples 10000
``` 