# Text Classification Experiments (Project5)

A PyTorch research project comparing character-level VDCNN and DenseNet models on Yelp Reviews and Yahoo Answers.

## 📖 Project Goal & Motivation

This project implements VDCNN and DenseNet text-classification models, with YAML experiment settings, training and evaluation code, and research documentation. It explores the models described in the referenced papers; an exact reproduction of their results has not been independently verified.

Accuracy values in [FINAL_REPORT.md](FINAL_REPORT.md) are historical report claims. The repository does not include the corresponding training logs or model checkpoints needed to reproduce those values, and they have not been revalidated during this documentation update.

## 🏗️ Architecture & Technical Highlights

*   **Experiment Configuration**: YAML files specify training parameters, with model and dataset choices also supplied to the training command.
*   **Model Implementations**: Contains character-level VDCNN and DenseNet implementations, plus a model factory and separate training/evaluation modules.
*   **Research Materials**: Includes dataset instructions, experiment reports, and unit/integration test source files.

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![YAML](https://img.shields.io/badge/YAML-CB171E?style=for-the-badge&logo=yaml&logoColor=white)
![Pytest](https://img.shields.io/badge/pytest-0A9B71?style=for-the-badge&logo=pytest)

## 🚀 Installation & Usage

Follow these steps to set up and run the project:

### 1. Clone the Repository

```bash
git clone https://github.com/Jackela/Project5.git
cd Project5
```

### 2. Create and Activate a Virtual Environment

*   **On Windows:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
*   **On macOS/Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

### 3. Install Dependencies

```bash
pip install -r p5_text_classification/requirements.txt
```

### 4. Run an Experiment

Run the following from the repository root after placing the Yelp dataset files at `p5_text_classification/data/yelp/train.csv` and `p5_text_classification/data/yelp/test.csv`. Dataset CSV files are not bundled in the repository. The training entry point requires all three arguments: `--model`, `--dataset`, and `--config`; relative configuration paths are resolved from the repository root.

```bash
python p5_text_classification/src/train.py --model vdcnn --dataset yelp --config p5_text_classification/configs/training/yelp_vdcnn.yaml
```

Use `--dataset yahoo` with `p5_text_classification/configs/training/yahoo_vdcnn.yaml` for the corresponding Yahoo experiment. Refer to [DATASETS.md](DATASETS.md) for dataset preparation. [USAGE.md](USAGE.md) contains additional historical usage notes; the command above reflects the current `train.py` argument parser.

AI-assisted maintenance currently covers documentation and source checks. Training and application tests were not run for this README update.

## 📄 License

This project is licensed under the MIT License.

---

**MIT License**

Copyright (c) 2024 Weixuan Kong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
