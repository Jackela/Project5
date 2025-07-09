# Presentation Charts and Diagrams

This document contains all the necessary charts and diagrams for the project presentation, with titles and descriptions in English.

## 1. System Architecture Diagram

This diagram illustrates the high-level architecture of the text classification framework, showcasing the interaction between core components. It highlights the modular design and adherence to SOLID principles.

```mermaid
graph TD
    subgraph "Input Layer"
        A[CLI / User Input] --> B(main.py);
    end

    subgraph "Configuration Core"
        B --> C{ConfigManager};
        C --> D1[global_defaults.yaml];
        C --> D2[dataset.yaml];
        C --> D3[model.yaml];
        C --> D4[CLI Overrides];
    end

    subgraph "Data Pipeline"
        E[Raw CSV Data] --> F[TextPreprocessor];
        F --> G(torch.utils.data.DataLoader);
    end

    subgraph "Model Core"
        H{Model Factory} --> I1[VDCNN];
        H --> I2[DenseNet];
        C --> H;
    end

    subgraph "Training & Evaluation Engine"
        J[Trainer] --> K[Evaluator];
        G --> J;
        H --> J;
        C --> J;
    end

    subgraph "Output & Reporting"
        L[Logger] --> M[Log Files (.log)];
        J --> N[Training History (.csv)];
        K --> O[Evaluation Metrics];
        P[analyze_logs.py] --> Q[Charts & Reports (.png, .md)];
        N --> P;
        M --> P;
    end

    B --> C;
    B --> F;
    B --> H;
    B --> J;
    J --> L;
    K --> L;
```

## 2. Data Flow Diagram

This diagram describes the end-to-end data flow, from raw data ingestion to the final output of predictions and performance metrics.

```mermaid
sequenceDiagram
    participant User
    participant main.py
    participant TextPreprocessor
    participant DataLoader
    participant Model
    participant Trainer
    participant Logger

    User->>main.py: Executes with CLI args (e.g., --model densenet)
    main.py->>TextPreprocessor: Initializes with character set
    main.py->>DataLoader: Loads raw CSV data and wraps with Dataset
    
    loop For each epoch
        Trainer->>DataLoader: Requests a batch of data
        DataLoader->>TextPreprocessor: Encodes text batch to integer sequences
        TextPreprocessor-->>DataLoader: Returns encoded tensors
        DataLoader-->>Trainer: Provides batch of (data_tensor, label_tensor)
        
        Trainer->>Model: model(data_tensor)
        Model-->>Trainer: Returns logits
        
        Trainer->>Trainer: Calculates loss (CrossEntropy)
        Trainer->>Trainer: Backpropagation & optimizer.step()
        Trainer->>Logger: Logs training progress (batch loss, etc.)
    end

    Trainer-->>main.py: Returns final training history
    main.py->>main.py: Saves history to CSV file
    main.py-->>User: Prints completion summary
``` 

## 3. Model Architecture Comparison

This diagram highlights the key architectural differences between the VDCNN and DenseNet models as implemented in this project. It contrasts VDCNN's sequential convolutional blocks with DenseNet's feature-reuse-centric design.

```mermaid
classDiagram
    direction LR

    class ModelInterface {
        <<interface>>
        +create_model() nn.Module
        +get_model_info() dict
    }

    class VDCNN {
        +depth: int
        +vocabulary_size: int
        +num_classes: int
        -conv_blocks: nn.ModuleList
        +forward(x) Tensor
    }
    
    class DenseNet {
        +growth_rate: int
        +block_config: tuple
        -dense_blocks: nn.ModuleList
        -transition_layers: nn.ModuleList
        +forward(x) Tensor
    }

    class _ConvBlock {
        <<VDCNN Component>>
        -layers: int
        -conv1: nn.Conv1d
        -bn1: nn.BatchNorm1d
        -conv2: nn.Conv1d
        -bn2: nn.BatchNorm1d
        -pooling: nn.MaxPool1d
    }

    class _DenseBlock {
        <<DenseNet Component>>
        -num_layers: int
        +forward(x) Tensor
    }
    
    class _DenseLayer {
        <<DenseNet Component>>
        -growth_rate: int
        +forward(x) Tensor
    }

    class _Transition {
        <<DenseNet Component>>
        -compression: float
        +forward(x) Tensor
    }

    ModelInterface <|-- VDCNN
    ModelInterface <|-- DenseNet
    VDCNN o-- _ConvBlock : contains
    DenseNet o-- _DenseBlock : contains
    DenseNet o-- _Transition : contains
    _DenseBlock o-- _DenseLayer : contains
``` 

## 4. Project Development Timeline

This Gantt chart outlines the major milestones and development phases of the project, showcasing the progression from initial setup to the final, robust experimentation framework.

```mermaid
gantt
    title Project Development Timeline
    dateFormat  YYYY-MM-DD
    axisFormat %Y-%m-%d
    
    section Project Setup & Initial Model
    Initial VDCNN Implementation :done, p1, 2025-06-20, 3d
    Configuration System Design :done, p2, after p1, 4d
    
    section DenseNet Integration & Automation
    DenseNet Model Implementation :done, p3, after p2, 4d
    Automation Scripting (Quick & Full Tests) :done, p4, after p3, 5d
    Log Analysis & Reporting Scripts :done, p5, after p4, 3d

    section Refactoring & Internationalization
    Full Code Internationalization :done, p6, after p5, 3d
    Script Robustness Enhancement :done, p7, after p6, 2d
    
    section Finalization & Reporting
    Log & History Export Refinement :active, p8, after p7, 2d
    Final Report Generation :p9, after p8, 2d

``` 