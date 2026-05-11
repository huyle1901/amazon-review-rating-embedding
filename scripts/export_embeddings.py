import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.model import build_sentiment_model


def parse_args():
    parser = argparse.ArgumentParser(description="Export learned word embeddings to TensorBoard.")
    parser.add_argument("--artifact-dir", type=str, default="artifacts/amazon_reviews_advanced")
    parser.add_argument("--log-dir", type=str, default="runs/sentiment_embeddings")
    parser.add_argument("--top-k", type=int, default=3000)
    return parser.parse_args()


def main():
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    log_dir = args.log_dir

    # Load config
    with open(artifact_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    # Load vocab
    with open(artifact_dir / "vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Rebuild model architecture
    model = build_sentiment_model(
        model_type=config.get("model_type", "embedding_dnn"),
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        padding_idx=0,
        cnn_num_filters=config.get("cnn_num_filters", 128),
        cnn_kernel_sizes=config.get("cnn_kernel_sizes", "3,5,7"),
        lstm_hidden_dim=config.get("lstm_hidden_dim", 128),
        lstm_layers=config.get("lstm_layers", 2),
        num_classes=config.get("num_classes", 1),
    )

    # Load trained weights
    model.load_state_dict(
        torch.load(artifact_dir / "model.pt", map_location="cpu")
    )
    model.eval()

    # Get embedding matrix
    embedding_matrix = model.embedding.weight.detach().cpu()

    # Convert vocab: word -> id thành id -> word
    id_to_word = {idx: word for word, idx in vocab.items()}

    # Không nên visualize toàn bộ nếu vocab quá lớn
    top_k = min(args.top_k, embedding_matrix.shape[0])

    selected_ids = list(range(top_k))
    selected_embeddings = embedding_matrix[selected_ids]
    metadata = [id_to_word[i] for i in selected_ids]

    writer = SummaryWriter(log_dir)

    writer.add_embedding(
        mat=selected_embeddings,
        metadata=metadata,
        tag="word_embeddings",
    )

    writer.close()

    print(f"Loaded artifacts from {artifact_dir}")
    print(f"Exported {top_k} word embeddings to {log_dir}")
    print("Run: uv run tensorboard --logdir runs")


if __name__ == "__main__":
    main()
