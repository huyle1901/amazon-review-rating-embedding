from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    dataset_name: str = "SetFit/amazon_reviews_multi_en"
    text_column: str = "text"
    label_column: str = "label"
    train_split: str = "train"
    validation_split: str = "test"
    output_dir: Path = Path("artifacts/amazon_reviews_advanced")

    num_classes: int = 5
    max_vocab_size: int = 50_000
    min_freq: int = 1
    max_length: int = 256

    model_type: str = "hybrid_cnn_lstm_attention"
    embedding_dim: int = 200
    hidden_dim: int = 256
    cnn_num_filters: int = 128
    cnn_kernel_sizes: str = "3,5,7"
    lstm_hidden_dim: int = 128
    lstm_layers: int = 2
    dropout: float = 0.5

    batch_size: int = 32
    epochs: int = 15
    learning_rate: float = 5e-4
    weight_decay: float = 1e-3
    early_stopping_patience: int = 4
    clip_grad_norm: float = 1.0

    seed: int = 42
