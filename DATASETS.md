# 📊 Dataset Specifications

## 🎯 Overview

This project uses two benchmark datasets widely applied in the text classification domain: **Yahoo Answers** and **Yelp Review Polarity**. Both datasets were compiled by Xiang Zhang et al. in their paper "Character-level Convolutional Networks for Text Classification," making them highly suitable for reproducing and validating character-level deep learning models.

## 📚 Yahoo Answers Dataset

### Academic Background
This dataset is a common benchmark for multi-class classification tasks, used to evaluate a model's ability to classify text across a diverse and balanced set of topics. Its 10 categories cover a wide range of social, scientific, and lifestyle subjects.

### Basic Information
- **Full Name**: Yahoo! Answers Topic Classification Dataset
- **Version**: Version 2 (Updated 09/09/2015)
- **Type**: Multi-class classification (10 categories)
- **Language**: English
- **Source**: Yahoo! Research Alliance Webscope program

### Data Statistics
| Item                   | Count       |
| ---------------------- | ----------- |
| **Training Samples**   | 1,400,000   |
| **Testing Samples**    | 60,000      |
| **Number of Classes**  | 10          |
| **Training Samples/Class** | 140,000 |
| **Testing Samples/Class**  | 6,000   |

### Category Descriptions
1.  **Society & Culture**
2.  **Science & Mathematics**
3.  **Health**
4.  **Education & Reference**
5.  **Computers & Internet**
6.  **Sports**
7.  **Business & Finance**
8.  **Entertainment & Music**
9.  **Family & Relationships**
10. **Politics & Government**

### Data Format
**CSV format, 4 columns**:
`class_index,question_title,question_content,best_answer`

**Example Data**:
```csv
1,"How to cook pasta?","I want to learn cooking pasta properly","Boil water, add salt, cook for 8-12 minutes until al dente"
2,"What is calculus?","Can someone explain calculus basics?","Calculus is the mathematical study of continuous change..."
3,"Best exercise for weight loss?","What exercise burns most calories?","High-intensity interval training (HIIT) is very effective..."
```

### Feature Analysis
- **Average Question Length**: ~50-100 characters
- **Average Answer Length**: ~200-500 characters
- **Text Characteristics**:
  - Includes question title, content, and best answer.
  - Language is relatively formal.
  - Covers a broad range of topics.
  - Answer quality is relatively high.

## 🍽️ Yelp Review Dataset

### Academic Background
This dataset is a classic benchmark for sentiment analysis (specifically binary sentiment classification). It is sourced from real user reviews, features colloquial text styles, and is suitable for evaluating a model's performance on informal, emotionally charged text.

### Basic Information
- **Full Name**: Yelp Review Polarity Dataset
- **Version**: Version 1 (Updated 09/09/2015)
- **Type**: Binary classification (sentiment analysis)
- **Language**: English
- **Source**: Yelp Dataset Challenge 2015

### Data Statistics
| Item                   | Count     |
| ---------------------- | --------- |
| **Training Samples**   | 560,000   |
| **Testing Samples**    | 38,000    |
| **Number of Classes**  | 2         |
| **Negative Reviews (Train)** | 280,000 |
| **Positive Reviews (Train)** | 280,000 |
| **Negative Reviews (Test)**  | 19,000  |
| **Positive Reviews (Test)**  | 19,000  |

### Category Descriptions
1.  **Negative Sentiment (Class 1)**: 1-2 star reviews
2.  **Positive Sentiment (Class 2)**: 3-4 star reviews

### Data Format
**CSV format, 2 columns**:
`sentiment,review_text`

**Example Data**:
```csv
1,"The food was terrible and service was slow. Would not recommend this place."
2,"Amazing restaurant! Great food and friendly staff. Will definitely come back!"
1,"Overpriced and disappointing. The pasta was cold and the sauce was bland."
2,"Best pizza in town! Fresh ingredients and perfect crust. Highly recommended!"
```

### Feature Analysis
- **Average Review Length**: ~150-300 characters
- **Text Characteristics**:
  - Real user reviews, with colloquial language.
  - Contains strongly emotional vocabulary.
  - Covers aspects like restaurants, service, food quality.
  - Sentiment polarity is quite distinct.

## 🔧 Preprocessing Pipeline

### Character-level Processing
The project employs character-level text processing instead of traditional word-level methods.

#### Alphabet
```python
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\_@#$%^&*~+=<>()[]{}"
```
- **Characters Included**: 26 lowercase letters + 10 digits + 33 special symbols
- **Total Count**: 69 characters
- **Special Tokens**: `<PAD>` (ID: 0), `<UNK>` (ID: 70)

#### Preprocessing Steps

1.  **Text Cleaning**
    ```python
    def clean_text(text: str) -> str:
        text = text.lower()  # Convert to lowercase
        # Replace characters not in ALPHABET with a space
        cleaned_text = ''.join([char if char in ALPHABET else ' ' for char in text])
        return cleaned_text
    ```

2.  **Character Encoding**
    ```python
    def tokenize_and_encode(text: str, char_to_id: dict) -> list[int]:
        cleaned_text = clean_text(text)
        # Convert characters to integer IDs
        encoded_text = [char_to_id.get(char, UNK_TOKEN_ID) for char in cleaned_text]
        
        # Pad or truncate to a fixed length
        if len(encoded_text) < SEQUENCE_LENGTH:
            encoded_text += [PAD_TOKEN_ID] * (SEQUENCE_LENGTH - len(encoded_text))
        else:
            encoded_text = encoded_text[:SEQUENCE_LENGTH]
        
        return encoded_text
    ```

3.  **Sequence Length**
    - **Fixed Length**: 1014 characters
    - **Padding**: Short texts are padded with PAD_TOKEN_ID (0).
    - **Truncation**: Long texts are truncated from the end.

### Data Loading Pipeline

#### Yahoo Dataset Handling
```python
def load_yahoo_data(file_path):
    df = pd.read_csv(file_path, header=None)
    df.columns = ['class_index', 'question_title', 'question_content', 'answer']
    
    # Combine all text fields
    texts = (df['question_title'].fillna('') + ' ' + 
             df['question_content'].fillna('') + ' ' + 
             df['answer'].fillna('')).values
    
    # Convert labels from 1-based to 0-based
    labels = df['class_index'].values - 1
    
    return texts, labels
```

#### Yelp Dataset Handling
```python
def load_yelp_data(file_path):
    df = pd.read_csv(file_path)
    texts = df['review'].values
    labels = df['sentiment'].values # Already 0-based
    
    return texts, labels
```

## 📈 Dataset Statistical Analysis

### Text Length Distribution

#### Yahoo Dataset
- **Shortest Text**: ~10 characters
- **Longest Text**: ~2000+ characters
- **Mean Length**: ~300 characters
- **Median Length**: ~250 characters

#### Yelp Dataset
- **Shortest Text**: ~5 characters
- **Longest Text**: ~1500+ characters
- **Mean Length**: ~150 characters
- **Median Length**: ~120 characters

### Character Distribution Characteristics

**Common Character Frequencies** (sorted by frequency):
1.  Space ' '
2.  Letters 'e', 'a', 't', 'i', 'n', 'o', 's'
3.  Punctuation '.', ',', '?', '!'
4.  Digits '0'-'9'

**Special Character Handling**:
-   HTML escape characters are cleaned.
-   Emojis are replaced with spaces.
-   Non-ASCII characters are filtered out.

## 💡 Usage Recommendations

### For Beginners
1.  **Start with Yelp**: The binary classification task is simpler.
2.  **Use a Small Sample**: Test with `--max_samples 1000` first.
3.  **Observe the Effects**: Understand how preprocessing impacts results.

### For Research
1.  **Yahoo is More Challenging**: The 10-class task requires more sophisticated feature extraction.
2.  **Character-level Advantage**: More robust against spelling errors and new words.
3.  **Comparative Experiments**: Validate model generalization across both datasets.

## 🔍 Data Quality Characteristics

### Yahoo Dataset Advantages
- ✅ Large dataset, balanced classes
- ✅ High-quality text
- ✅ Broad topic coverage
- ✅ Suitable for multi-class research

### Yahoo Dataset Challenges
- ❗ Potential topic overlap between classes
- ❗ Q&A format may contain noise
- ❗ Requires stronger feature extraction capabilities

### Yelp Dataset Advantages
- ✅ Clear sentiment polarity
- ✅ Authentic user-generated content
- ✅ Good for starting with sentiment analysis
- ✅ Simpler binary classification task

### Yelp Dataset Challenges
- ❗ Colloquial user language
- ❗ May contain spelling errors
- ❗ Diverse ways of expressing sentiment

These datasets provide a solid foundation for text classification research and are highly valuable for both academic and industrial applications. 📊 