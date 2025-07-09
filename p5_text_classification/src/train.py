import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import logging
from datetime import datetime
from models.vdcnn_model import VDCNN
from models.densenet import DenseNet
from data.text_preprocessor import tokenize_and_encode, create_char_to_id_mapping, SEQUENCE_LENGTH
from torch.optim.lr_scheduler import MultiStepLR
from pathlib import Path
import yaml

# Define project root and REPORTS directory
BASE_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR = BASE_DIR / 'REPORTS'
REPORTS_DIR.mkdir(exist_ok=True)

def setup_logger(model_name: str, dataset_name: str, depth: int = None):
    """
    Setup logger with both console and file output.
    
    Args:
        model_name: Name of the model (e.g., 'vdcnn', 'densenet')
        dataset_name: Name of the dataset (e.g., 'yahoo', 'yelp')
    
    Returns:
        Configured logger instance
    """
    # Create REPORTS logs directory if it doesn't exist
    log_dir = REPORTS_DIR / 'logs' / model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_name == 'vdcnn' and depth:
        log_filename = str(log_dir / f"{model_name}_{dataset_name}_depth{depth}_{timestamp}.log")
    else:
        log_filename = str(log_dir / f"{model_name}_{dataset_name}_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers to avoid duplication
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create and configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create and configure file handler
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"🗂️  Log file created: {log_filename}")
    logger.info("=" * 80)
    
    return logger

def load_data(dataset_path: str, batch_size: int, max_samples: int = None, num_workers: int = 0, logger=None):
    """
    Loads the dataset from a CSV file and creates a DataLoader.

    Args:
        dataset_path: The path to the dataset file.
        batch_size: The batch size for the DataLoader.
        max_samples: Maximum number of samples to load from the dataset. If None, loads all samples.
        num_workers: Number of worker processes for data loading.
        logger: Logger instance for output. If None, uses print statements.

    Returns:
        A DataLoader for the dataset and the number of classes.
    """
    log = logger.info if logger else print
    
    log(f"📁 Loading data from: {dataset_path}")
    log(f"⚙️  Max samples: {max_samples if max_samples else 'All'}")
    
    # Determine dataset type based on path
    if 'yahoo' in dataset_path:
        log("📊 Dataset type: Yahoo Answers")
        df = pd.read_csv(dataset_path, header=None, nrows=max_samples)
        df.columns = ['class_index', 'question_title', 'question_content', 'answer']
        texts = (df['question_title'].fillna('') + ' ' + df['question_content'].fillna('') + ' ' + df['answer'].fillna('')).values
        labels = df['class_index'].values - 1 # Yahoo labels are 1-based, convert to 0-based
        log(f"✅ Loaded {len(texts)} Yahoo samples")
    elif 'yelp' in dataset_path:
        log("📊 Dataset type: Yelp Reviews")
        # Yelp CSV has no header: sentiment,label and review text
        df = pd.read_csv(dataset_path, header=None, nrows=max_samples)
        df.columns = ['sentiment', 'review']
        # Convert sentiment labels (1,2) to 0-based (0,1)
        df['sentiment'] = df['sentiment'] - 1
        texts = df['review'].values
        labels = df['sentiment'].values
        log(f"✅ Loaded {len(texts)} Yelp samples")
    else:
        raise ValueError("Unsupported dataset type. Path must contain 'yahoo' or 'yelp'.")

    log("🔤 Creating character-to-ID mapping...")
    char_to_id = create_char_to_id_mapping()
    # Print actual character count (excluding PAD and UNK)
    log(f"📝 Character vocabulary size: {len(char_to_id) - 2}")
    
    log("🔢 Encoding texts to character sequences...")
    encoded_texts = [tokenize_and_encode(text, char_to_id) for text in texts]
    log(f"✅ Encoded {len(encoded_texts)} text samples")
    
    # Convert labels to integers if they are not already
    if not pd.api.types.is_numeric_dtype(labels):
        label_map = {label: i for i, label in enumerate(pd.unique(labels))}
        labels = [label_map[label] for label in labels]

    # Convert to tensors
    log("🔗 Converting to PyTorch tensors...")
    tensor_texts = torch.tensor(encoded_texts, dtype=torch.long)
    tensor_labels = torch.tensor(labels, dtype=torch.long)
    log(f"📏 Text tensor shape: {tensor_texts.shape}")
    log(f"🏷️  Label tensor shape: {tensor_labels.shape}")

    dataset = TensorDataset(tensor_texts, tensor_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    log(f"🚀 Created DataLoader with batch size: {batch_size}")

    num_classes = len(pd.unique(labels))
    log(f"🎯 Number of classes: {num_classes}")
    return dataloader, char_to_id, num_classes

def train(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device, logger=None):
    """
    Trains the model for one epoch.

    Args:
        model: The model to be trained.
        dataloader: The DataLoader for the training data.
        criterion: The loss function.
        optimizer: The optimizer.
        device: The device to train on (CPU or GPU).
        logger: Logger instance for output. If None, uses print statements.
    """
    log = logger.info if logger else print
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    num_batches = len(dataloader)
    log(f"🚂 Training on {num_batches} batches...")
    
    for batch_idx, (texts, labels) in enumerate(dataloader):
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Print progress every 10% of batches
        if (batch_idx + 1) % max(1, num_batches // 10) == 0:
            log(f"  📈 Batch {batch_idx + 1}/{num_batches}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    avg_acc = 100 * correct / total
    log(f"✅ Training completed. Average loss: {avg_loss:.4f}, Average accuracy: {avg_acc:.2f}%")
    return avg_loss, avg_acc

def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device, logger=None):
    """
    Evaluates the model on the validation set.

    Args:
        model: The model to be evaluated.
        dataloader: The DataLoader for the validation data.
        criterion: The loss function.
        device: The device to evaluate on (CPU or GPU).
        logger: Logger instance for output. If None, uses print statements.

    Returns:
        The validation loss and accuracy.
    """
    log = logger.info if logger else print
    
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    num_batches = len(dataloader)
    log(f"🔍 Evaluating on {num_batches} batches...")
    
    with torch.no_grad():
        for batch_idx, (texts, labels) in enumerate(dataloader):
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print progress every 20% of batches
            if (batch_idx + 1) % max(1, num_batches // 5) == 0:
                current_acc = 100 * correct / total
                log(f"  📊 Batch {batch_idx + 1}/{num_batches}, Acc so far: {current_acc:.2f}%")

    final_loss = total_loss / len(dataloader)
    final_acc = 100 * correct / total
    log(f"✅ Evaluation completed. Loss: {final_loss:.4f}, Accuracy: {final_acc:.2f}%")
    return final_loss, final_acc

def main():
    parser = argparse.ArgumentParser(description='Train a text classification model.')
    parser.add_argument('--model', type=str, required=True, choices=['vdcnn', 'densenet'], help='The model to train.')
    parser.add_argument('--dataset', type=str, required=True, choices=['yahoo', 'yelp'], help='The dataset to use.')
    parser.add_argument('--batch_size', type=int, default=128, help='The batch size.')
    parser.add_argument('--epochs', type=int, default=10, help='The number of epochs to train for.')
    parser.add_argument('--lr', type=float, default=0.01, help='The learning rate.')
    parser.add_argument('--depth', type=int, default=9, choices=[9, 17, 29, 49], help='The depth of the VDCNN model.')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to load from the dataset.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of worker processes for data loading.')
    parser.add_argument('--config', type=str, required=True, help='Relative path to experiment YAML config file.')
    args = parser.parse_args()
    # Load experiment configuration
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = BASE_DIR / args.config
    with open(config_path, 'r', encoding='utf-8') as cf:
        cfg = yaml.safe_load(cf)
    # Override CLI args with config file values
    args.dataset = cfg.get('dataset', args.dataset)
    args.model = cfg.get('model', {}).get('name', args.model)
    args.batch_size = cfg.get('training', {}).get('batch_size', args.batch_size)
    args.epochs = cfg.get('training', {}).get('epochs', args.epochs)
    args.lr = cfg.get('training', {}).get('lr', args.lr)
    # Support depth defined under model.params.depth or directly under model.depth in config
    model_cfg = cfg.get('model', {})
    args.depth = model_cfg.get('params', {}).get('depth', model_cfg.get('depth', args.depth))
    args.max_samples = cfg.get('training', {}).get('max_samples', args.max_samples)
    args.num_workers = cfg.get('training', {}).get('num_workers', args.num_workers)
    # Initialize best validation accuracy for model saving
    best_val_acc = 0.0

    # Setup logger first thing in main
    logger = setup_logger(args.model, args.dataset, args.depth if args.model == 'vdcnn' else None)
    
    # Setup device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️  Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    logger.info(f"{'='*50}")
    logger.info(f"🎯 TRAINING CONFIGURATION")
    logger.info(f"{'='*50}")
    logger.info(f"📊 Model: {args.model.upper()}")
    logger.info(f"📊 Dataset: {args.dataset.upper()}")
    logger.info(f"🔢 Batch size: {args.batch_size}")
    logger.info(f"📈 Epochs: {args.epochs}")
    logger.info(f"⚡ Learning rate: {args.lr}")
    logger.info(f"📏 Max samples: {args.max_samples if args.max_samples else 'All'}")
    logger.info(f"👥 Workers: {args.num_workers}")
    if args.model == 'vdcnn':
        logger.info(f"🏗️  VDCNN depth: {args.depth}")
    logger.info(f"{'='*50}")

    # Load data - Smart path detection
    def find_data_path(dataset_name, split):
        """Smart path detection for different environments"""
        possible_paths = [
            f'p5_text_classification/data/{dataset_name}/{split}.csv',  # Original path
            f'data/{dataset_name}/{split}.csv',  # If running from project root
            f'{dataset_name}/{split}.csv',  # If running from data directory
            f'/content/drive/MyDrive/p5_text_classification/data/{dataset_name}/{split}.csv',  # Colab path
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"✅ Found data file: {path}")
                return path
        
        logger.error(f"❌ Data file not found. Tried paths:")
        for path in possible_paths:
            logger.error(f"   - {path}")
        raise FileNotFoundError(f"Could not find {dataset_name}/{split}.csv in any expected location")
    
    train_path = find_data_path(args.dataset, 'train')
    test_path = find_data_path(args.dataset, 'test')
    
    logger.info("📚 LOADING TRAINING DATA")
    logger.info("-" * 30)
    train_loader, char_to_id, num_classes = load_data(train_path, args.batch_size, args.max_samples, args.num_workers, logger)
    
    logger.info("📚 LOADING TEST DATA")
    logger.info("-" * 30)
    test_loader, _, _ = load_data(test_path, args.batch_size, args.max_samples, args.num_workers, logger)

    # Initialize model
    logger.info(f"🧠 INITIALIZING MODEL: {args.model.upper()}")
    logger.info("-" * 30)
    if args.model == 'vdcnn':
        model = VDCNN(num_classes=num_classes, vocab_size=len(char_to_id), embedding_dim=16, depth=args.depth).to(device)
        logger.info(f"✅ VDCNN model created (depth={args.depth}, classes={num_classes})")
    elif args.model == 'densenet':
        # Use the character-to-ID mapping size as the embedding vocabulary size
        vocab_size = len(char_to_id)
        model = DenseNet(num_classes=num_classes, vocab_size=vocab_size).to(device)
        logger.info(f"✅ DenseNet model created (classes={num_classes}, vocab_size={vocab_size})")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 Total parameters: {total_params:,}")
    logger.info(f"📊 Trainable parameters: {trainable_params:,}")

    # Define loss function and optimizer
    logger.info(f"⚙️  SETTING UP TRAINING")
    logger.info("-" * 30)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    logger.info(f"✅ Loss function: CrossEntropyLoss")
    logger.info(f"✅ Optimizer: SGD (lr={args.lr}, momentum=0.9, weight_decay=1e-4)")

    # Configure learning rate scheduler based on experiment config
    sched_cfg = cfg['training']['scheduler']
    if 'milestones' in sched_cfg:
        milestones = sched_cfg['milestones']
    else:
        ratios = sched_cfg.get('milestones_ratio', [])
        milestones = [max(1, int(args.epochs * r)) for r in ratios]
    gamma = sched_cfg.get('gamma', 0.1)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    logger.info(f"✅ Learning rate scheduler: MultiStepLR (milestones={milestones}, gamma={gamma})")

    # Train and evaluate
    logger.info(f"🚀 STARTING TRAINING")
    logger.info("=" * 50)
    
    training_history = []
    for epoch in range(args.epochs):
        logger.info(f"🔥 EPOCH {epoch+1}/{args.epochs}")
        logger.info("=" * 30)
        
        logger.info(f"🏋️  Training phase...")
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, logger)
        
        # Update learning rate at the end of each epoch
        scheduler.step()
        logger.info(f"🔄 Learning rate for next epoch: {scheduler.get_last_lr()[0]}")

        logger.info(f"📊 Evaluation phase...")
        val_loss, val_acc = evaluate(model, test_loader, criterion, device, logger)
        
        # Append epoch results to history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc
        })

        logger.info(f"🎯 EPOCH {epoch+1} RESULTS:")
        logger.info(f"   📉 Validation Loss: {val_loss:.4f}")
        logger.info(f"   🎯 Validation Accuracy: {val_acc:.2f}%")
        # Save model if it's the best so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = REPORTS_DIR / f"{args.model}_{args.dataset}_depth{args.depth}_best.pth"
            torch.save(model.state_dict(), model_path)
            logger.info(f"💾 Saved new best model to {model_path}")

    # Save training history
    history_df = pd.DataFrame(training_history)
    history_dir = REPORTS_DIR / 'training_history'
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / f"{args.model}_{args.dataset}_depth{args.depth}_history.csv"
    history_df.to_csv(history_path, index=False)
    logger.info(f"📜 Saved training history to {history_path}")

    logger.info(f"�� TRAINING COMPLETED!")
    logger.info("=" * 50)
    
    # Calculate and log total training time
    end_time = datetime.now()
    total_training_time = (end_time - training_start_time).total_seconds()
    logger.info(f"⏳ Total training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")

    # Log final summary
    logger.info("📋 TRAINING SUMMARY:")
    logger.info(f"   🏆 Final validation accuracy: {val_acc:.2f}%")
    logger.info(f"   📊 Model: {args.model.upper()}")
    logger.info(f"   📊 Dataset: {args.dataset.upper()}")
    logger.info(f"   📈 Epochs trained: {args.epochs}")
    logger.info(f"   📊 Total parameters: {total_params:,}")

    # Generate predictions on test set
    logger.info("🔖 Generating predictions on test set...")
    true_labels = []
    predicted_labels = []
    model.eval()
    with torch.no_grad():
        for texts, labels in test_loader:
            texts = texts.to(device)
            outputs = model(texts)
            _, preds = torch.max(outputs, 1)
            true_labels.extend(labels.tolist())
            predicted_labels.extend(preds.cpu().tolist())
    preds_df = pd.DataFrame({'true_label': true_labels, 'predicted_label': predicted_labels})
    preds_path = REPORTS_DIR / f"{args.model}_{args.dataset}_depth{args.depth}_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    logger.info(f"✅ Saved predictions to {preds_path}")

if __name__ == '__main__':
    main()