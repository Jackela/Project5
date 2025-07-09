#!/usr/bin/env python3
"""
Generate confusion matrix heatmaps for error analysis.
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import argparse
from pathlib import Path

# Mapping from label indices to human-readable class names
YELP_MAPPING = {
    0: 'Negative',
    1: 'Positive'
}

YAHOO_MAPPING = {
    0: 'Society & Culture',
    1: 'Science & Mathematics',
    2: 'Health',
    3: 'Education & Reference',
    4: 'Computers & Internet',
    5: 'Sports',
    6: 'Business & Finance',
    7: 'Entertainment & Music',
    8: 'Family & Relationships',
    9: 'Politics & Government'
}

def generate_confusion_matrix(input_path: str, mapping: dict, output_path: str) -> None:
    """
    Load predictions CSV and generate a confusion matrix heatmap.

    Args:
        input_path: Path to the CSV file with true_label and predicted_label.
        mapping: Dictionary mapping label indices to class names.
        output_path: Path where the heatmap PNG will be saved.
    """
    df = pd.read_csv(input_path)
    y_true = df['true_label']
    y_pred = df['predicted_label']
    labels = sorted(mapping.keys())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[mapping[i] for i in labels], columns=[mapping[i] for i in labels])

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f'Confusion Matrix - {os.path.splitext(os.path.basename(output_path))[0]}')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    """
    Main function to generate a confusion matrix from a given predictions file.
    """
    parser = argparse.ArgumentParser(description="Generate a confusion matrix from a predictions file.")
    parser.add_argument('--predictions', type=str, required=True, help="Path to the prediction CSV file.")
    parser.add_argument('--dataset', type=str, required=True, choices=['yahoo', 'yelp'], help="Dataset type for correct class labels.")
    parser.add_argument('--output', type=str, required=True, help="Path for the output confusion matrix PNG.")
    args = parser.parse_args()

    mapping = YAHOO_MAPPING if args.dataset == 'yahoo' else YELP_MAPPING
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating confusion matrix for: {args.predictions}")
    generate_confusion_matrix(args.predictions, mapping, output_path)
    print(f"✅ Confusion matrix saved to {output_path}")

if __name__ == '__main__':
    main() 