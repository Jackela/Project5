# **Project 5 Final Report: CNNs for Text Classification**

## **1. Executive Summary**
This report details a comparative study of two Convolutional Neural Network (CNN) architectures, **VDCNN** (Very Deep CNN) and **DenseNet**, for text classification. We evaluated these models on two datasets of varying complexity: **Yelp Reviews Polarity** and **Yahoo Answers**. Key findings indicate that while both models are highly effective, the deeper VDCNN architecture generally outperforms DenseNet, particularly on the more complex, 10-class Yahoo Answers dataset. A bonus experiment investigating the impact of model depth (VDCNN-9 vs. VDCNN-17) revealed that increased depth marginally improves accuracy but at a significant computational cost, suggesting a point of diminishing returns.

## **2. Methodology**
### Architectures
- **VDCNN:** A deep CNN architecture that operates directly at the character level, using a sequence of small convolutions (3x1) and temporal pooling layers. This allows it to learn hierarchical features from raw text without needing word embeddings. We tested depths of 9 and 17 convolutional layers.
- **DenseNet:** A character-level CNN inspired by the DenseNet architecture from computer vision. It uses dense blocks where each layer is connected to every other layer in a feed-forward fashion, promoting feature reuse and gradient flow.

### Datasets
- **Yelp Reviews Polarity:** A binary classification task with 560,000 training samples and 38,000 testing samples. The task is to classify reviews as positive or negative.
- **Yahoo Answers:** A more complex 10-class classification task with 1,400,000 training samples and 60,000 testing samples, covering diverse topics like science, sports, and business.

### Training Configuration
All experiments were run with a consistent training configuration to ensure fair comparison. Key parameters included the SGD optimizer (learning rate 0.01, momentum 0.9), a batch size of 128, and a Multi-Step learning rate scheduler that reduced the learning rate at 50% and 75% of the total epochs.

## **3. Core Experimental Results**
### **3.1. Performance Comparison**
The final validation accuracies of the four core experiments are summarized below. VDCNN demonstrates superior performance on both datasets, with a more pronounced advantage on the more challenging Yahoo Answers dataset.

![Core Accuracy Comparison](plots/core_accuracy_comparison.png)

| Model | Dataset | Final Accuracy |
| :--- | :--- | :--- |
| **VDCNN-9** | **Yelp** | **94.70%** |
| **VDCNN-9** | **Yahoo** | **71.85%** |
| DenseNet | Yelp | 90.41% |
| DenseNet | Yahoo | 62.47% |

### **3.2. Analysis**
- **Model Performance:** The VDCNN architecture consistently outperformed DenseNet. Its deeper, sequential structure appears more effective at capturing the complex, hierarchical nature of language from character-level inputs compared to DenseNet's feature-reuse-focused design.
- **Dataset Impact:** The performance gap between the models widened significantly on the Yahoo dataset. This suggests that the robustness of the VDCNN architecture is more apparent when dealing with a larger number of classes and more diverse linguistic content. The simpler binary sentiment classification of the Yelp dataset was an easier task for both models, resulting in a smaller performance difference.

## **4. Bonus: In-depth Exploration**
### **4.1. Error Analysis**
The confusion matrices provide insight into the models' classification patterns, particularly on the 10-class Yahoo dataset.

![VDCNN on Yelp](plots/confusion_matrices/vdcnn_yelp_confusion_matrix.png)
![DenseNet on Yelp](plots/confusion_matrices/densenet_yelp_confusion_matrix.png)
![VDCNN on Yahoo](plots/confusion_matrices/vdcnn_yahoo_confusion_matrix.png)
![DenseNet on Yahoo](plots/confusion_matrices/densenet_yahoo_confusion_matrix.png)

On the Yahoo dataset, both models struggled with semantically close categories. For instance, "Society & Culture" was often confused with "Family & Relationships," and "Science & Mathematics" with "Computers & Internet." However, VDCNN shows a more diagonally dominant matrix, indicating fewer gross misclassifications compared to DenseNet, which aligns with its higher overall accuracy.

### **4.2. Model Depth Investigation**
To explore the "very deep" nature of VDCNN, we compared the baseline 9-layer model with a deeper 17-layer variant on the Yahoo dataset.

![Depth vs. Performance](plots/depth_tradeoff_comparison.png)

![Depth Accuracy Curves](plots/depth_accuracy_curves.png)

- **Performance vs. Cost:** Increasing the depth from 9 to 17 layers yielded only a minor accuracy improvement (from 71.85% to 72.01%). However, this came at the cost of a significant increase in trainable parameters.
- **Training Dynamics:** The validation accuracy curves show that both models follow a similar learning trajectory. The deeper VDCNN-17 does not exhibit signs of overfitting compared to the shallower model within the trained epochs, but its marginal benefit suggests that for this task, a 9-layer architecture already captures most of the learnable features.

## **5. Conclusion**
This project successfully demonstrates the power of deep, character-level CNNs for text classification. The **VDCNN** architecture proved to be a robust and high-performing choice, outclassing DenseNet across both a simple binary task and a complex multi-class problem. Our investigation into model depth further revealed that while deeper models can offer slight performance gains, they quickly encounter diminishing returns, highlighting the critical importance of balancing model complexity with computational cost for practical applications. Future work could explore even deeper models with regularization techniques like Shake-Shake or investigate the efficiency of these architectures on mobile devices. 