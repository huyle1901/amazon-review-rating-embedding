import argparse
import random
import sys
from dataclasses import asdict
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.text_utils import build_vocab
from src.dataset import SentimentDataset
from src.model import build_sentiment_model
from configs.train_config import TrainConfig



def parse_args():
    parser = argparse.ArgumentParser(
        description="Train sentiment classification model using Word Embedding + DNN."
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--text-column", type=str, default=None)
    parser.add_argument("--label-column", type=str, default=None)
    parser.add_argument("--train-split", type=str, default=None)
    parser.add_argument("--validation-split", type=str, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-type", type=str, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--cnn-num-filters", type=int, default=None)
    parser.add_argument("--cnn-kernel-sizes", type=str, default=None)
    parser.add_argument("--lstm-hidden-dim", type=int, default=None)
    parser.add_argument("--lstm-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--max-vocab-size", type=int, default=None)
    parser.add_argument("--min-freq", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--clip-grad-norm", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)


    return parser.parse_args()

def override_config(config: TrainConfig, args) -> TrainConfig:
    if args.dataset_name is not None:
        config.dataset_name = args.dataset_name

    if args.text_column is not None:
        config.text_column = args.text_column

    if args.label_column is not None:
        config.label_column = args.label_column

    if args.train_split is not None:
        config.train_split = args.train_split

    if args.validation_split is not None:
        config.validation_split = args.validation_split

    if args.num_classes is not None:
        config.num_classes = args.num_classes

    if args.epochs is not None:
        config.epochs = args.epochs

    if args.batch_size is not None:
        config.batch_size = args.batch_size

    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate

    if args.embedding_dim is not None:
        config.embedding_dim = args.embedding_dim

    if args.hidden_dim is not None:
        config.hidden_dim = args.hidden_dim

    if args.dropout is not None:
        config.dropout = args.dropout

    if args.max_vocab_size is not None:
        config.max_vocab_size = args.max_vocab_size

    if args.min_freq is not None:
        config.min_freq = args.min_freq

    if args.max_length is not None:
        config.max_length = args.max_length

    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay

    if args.early_stopping_patience is not None:
        config.early_stopping_patience = args.early_stopping_patience

    if args.output_dir is not None:
        config.output_dir = Path(args.output_dir)

    if args.model_type is not None:
        config.model_type = args.model_type

    if args.seed is not None:
        config.seed = args.seed

    if args.cnn_num_filters is not None:
        config.cnn_num_filters = args.cnn_num_filters

    if args.cnn_kernel_sizes is not None:
        config.cnn_kernel_sizes = args.cnn_kernel_sizes

    if args.lstm_hidden_dim is not None:
        config.lstm_hidden_dim = args.lstm_hidden_dim

    if args.lstm_layers is not None:
        config.lstm_layers = args.lstm_layers

    if args.clip_grad_norm is not None:
        config.clip_grad_norm = args.clip_grad_norm

    return config

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, dataloader, criterion, device, num_classes: int):
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            if num_classes == 1:
                labels = labels.float()

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            if num_classes == 1:
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities > 0.5).long()
            else:
                predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.long().cpu().tolist())

    average_loss = total_loss / len(dataloader)

    accuracy = accuracy_score(all_labels, all_predictions)

    average = "binary" if num_classes == 1 else "macro"
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average=average,
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )
    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }

def save_training_history(history: List[Dict[str, float]], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)


def save_artifacts(model, vocab, config: TrainConfig, best_f1: float):
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = config.output_dir / "model.pt"
    vocab_path = config.output_dir / "vocab.json"
    config_path = config.output_dir / "config.json"

    torch.save(model.state_dict(), model_path)

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)

    saved_config = asdict(config)
    saved_config["output_dir"] = str(config.output_dir)
    saved_config["vocab_size"] = len(vocab)
    saved_config["best_f1"] = best_f1
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(saved_config, f, ensure_ascii=False, indent=4)

    print(f"Saved model: {model_path}")
    print(f"Saved vocab: {vocab_path}")
    print(f"Saved config: {config_path}")

def main():
    args = parse_args()
    config = TrainConfig()
    config = override_config(config, args)

    set_seed(config.seed)

    device = get_device()

    print("Training configuration:")
    print(config)
    print(f"Using device: {device}")

    # 1. Load dataset
    dataset = load_dataset(config.dataset_name)

    train_texts = dataset[config.train_split][config.text_column]
    train_labels = dataset[config.train_split][config.label_column]

    valid_texts = dataset[config.validation_split][config.text_column]
    valid_labels = dataset[config.validation_split][config.label_column]

    # 2. Build vocabulary
    vocab = build_vocab(
        texts=train_texts, 
        max_vocab_size=config.max_vocab_size, 
        min_freq=config.min_freq)

    print(f"Train samples: {len(train_texts)}")
    print(f"Validation samples: {len(valid_texts)}")
    print(f"Vocabulary size: {len(vocab)}")

    # 3. Create Datasets
    train_dataset = SentimentDataset(
        texts=train_texts,
        labels=train_labels,
        vocab=vocab,
        max_length=config.max_length,
    )

    valid_dataset = SentimentDataset(
        texts=valid_texts,
        labels=valid_labels,
        vocab=vocab,
        max_length=config.max_length,
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )


    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    # 5. Create model
    model = build_sentiment_model(
        model_type=config.model_type,
        vocab_size=len(vocab),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        padding_idx=vocab.get("<pad>", 0),
        cnn_num_filters=config.cnn_num_filters,
        cnn_kernel_sizes=config.cnn_kernel_sizes,
        lstm_hidden_dim=config.lstm_hidden_dim,
        lstm_layers=config.lstm_layers,
        num_classes=config.num_classes,
    ).to(device)

    if config.num_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
    )

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Model type: {config.model_type}")
    print(f"Trainable parameters: {trainable_params:,}")

    best_f1 = 0.0
    epochs_without_improvement = 0
    history = []

    # 6. Training loop
    for epoch in range(1, config.epochs + 1):
        model.train()
        
        total_train_loss = 0.0

        progress_bar = tqdm(train_loader,
                            desc = f"Epoch {epoch}/{config.epochs}",
                            )
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            if config.num_classes == 1:
                labels = labels.float()

            optimizer.zero_grad()

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            optimizer.step()

            total_train_loss += loss.item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss = total_train_loss / len(train_loader)

        valid_metrics = evaluate(
                model=model,
                dataloader=valid_loader,
                criterion=criterion,
                device=device,
                num_classes=config.num_classes,
            )

        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={valid_metrics['loss']:.4f} | "
            f"acc={valid_metrics['accuracy']:.4f} | "
            f"precision={valid_metrics['precision']:.4f} | "
            f"recall={valid_metrics['recall']:.4f} | "
            f"macro_f1={valid_metrics['f1']:.4f} | "
            f"weighted_f1={valid_metrics['weighted_f1']:.4f}"
        )

        epoch_history = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": valid_metrics["loss"],
            "val_accuracy": valid_metrics["accuracy"],
            "val_precision": valid_metrics["precision"],
            "val_recall": valid_metrics["recall"],
            "val_macro_f1": valid_metrics["f1"],
            "val_weighted_precision": valid_metrics["weighted_precision"],
            "val_weighted_recall": valid_metrics["weighted_recall"],
            "val_weighted_f1": valid_metrics["weighted_f1"],
        }
        history.append(epoch_history)
        save_training_history(history, config.output_dir)
        scheduler.step(valid_metrics["loss"])

    # 7, Save artifacts
        if valid_metrics["f1"] > best_f1:
            best_f1 = valid_metrics["f1"]
            epochs_without_improvement = 0
            save_artifacts(model, vocab, config, best_f1)
            print(f"New best model with F1: {best_f1:.4f}. Artifacts saved.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                print(
                    "Early stopping: "
                    f"F1 did not improve for {config.early_stopping_patience} epoch(s)."
                )
                break

    print("Training completed.")

if __name__ == "__main__":
    main()
