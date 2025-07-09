# Production-Grade Text Classification Framework (Project5)

Bridging academic research and robust software engineering for deep learning text classification.

## 📖 Project Goal & Motivation

This project had a dual objective: first, to accurately reproduce the results of the VDCNN (Very Deep Convolutional Neural Networks for Text Classification) paper in PyTorch; second, to meticulously re-architect the research prototype into a modern, production-ready framework. The core motivation was to demonstrate the ability to transform theoretical academic concepts into elegant, flexible, and functional code suitable for real-world AI engineering.

## 🏗️ Architecture & Technical Highlights

*   **Configuration-Driven Framework**: The entire framework is designed to be fully configuration-driven, utilizing **YAML** files to manage and define all aspects of experiments, from dataset paths and model architectures to training hyperparameters.
*   **SOLID Principles & Design Patterns**: The codebase was rigorously refactored following **SOLID principles**, promoting maintainability, flexibility, and extensibility. Advanced software design patterns were implemented, including the **Strategy Pattern** for swappable training logic and the **Factory Pattern with Dependency Injection (DI)** to dynamically construct components.
*   **Bridging Research & Engineering**: This project served as a critical lesson in bridging the gap between academic research and production-grade AI engineering, solidifying the ability to transform theoretical concepts into robust, functional code.

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![YAML](https://img.shields.io/badge/YAML-CB171E?style=for-the-badge&logo=yaml&logoColor=white)
![Pytest](https://img.shields.io/badge/pytest-0A9B71?style=for-the-badge&logo=pytest)

*Also built with a strong adherence to **SOLID Principles** and advanced software design patterns.*

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
pip install -r requirements.txt
```

### 4. Run an Experiment

Experiments are driven by YAML configuration files. To run a training experiment, specify a configuration file:

```bash
python p5_text_classification/src/train.py --config configs/training/your_experiment.yaml
```

Refer to `USAGE.md` and `DATASETS.md` for more detailed instructions on running experiments and dataset preparation.

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
