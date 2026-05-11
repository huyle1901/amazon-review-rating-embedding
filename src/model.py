import torch
import torch.nn as nn
import torch.nn.functional as F

class SentimentEmbeddingDNN(nn.Module):
    """
    Feedforward neural network for sentiment classification.

    Architecture:
        input_ids
        -> embedding layer
        -> mean pooling
        -> DNN classifier
        -> logit

    Output:
        logit: Single val   
    """
    def __init__(
        self, 
        vocab_size: int, 
        embedding_dim: int = 100, 
        hidden_dim: int = 128, 
        dropout: float = 0.3,
        num_classes: int = 1,
        padding_idx: int = 0,
        ):
        super(SentimentEmbeddingDNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids:
                shape [batch_size, seq_len]

            attention_mask:
                shape [batch_size, seq_len]
                1 = real token
                0 = padding token

        Returns:
            logits:
                shape [batch_size]
        """
        embeddded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]

        mask = attention_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]

        masked_embedded = embeddded * mask  # Zero out padding token embeddings

        sentence_vector = masked_embedded.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        logits = self.classifier(sentence_vector)

        if logits.size(-1) == 1:
            return logits.squeeze(-1)
        return logits
    
    def get_word_embedding_matrix(self) -> torch.Tensor:
        """
        Return learned  embedding matrix
        shape: [vocal_size, embedding_dim]
        """
        return self.embedding.weight.detach()
    
    def get_sentence_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the sentence embedding after mean pooling.
        """
        embedded = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1)
        masked_embedded = embedded * mask
        sentence_vector = masked_embedded.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return sentence_vector


class HybridCnnLstmAttentionSentimentModel(nn.Module):
    """
    Deeper sentiment model:
        embedding
        -> parallel TextCNN blocks for local n-gram features
        -> BiLSTM for contextual sequence features
        -> attention pooling for important token features
        -> multilayer classifier
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 200,
        hidden_dim: int = 256,
        cnn_num_filters: int = 128,
        cnn_kernel_sizes: str = "3,5,7",
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.5,
        num_classes: int = 1,
        padding_idx: int = 0,
    ):
        super().__init__()

        kernel_sizes = [int(size.strip()) for size in cnn_kernel_sizes.split(",")]

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embedding_dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=cnn_num_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            for kernel_size in kernel_sizes
        ])

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        feature_dim = embedding_dim + (cnn_num_filters * len(kernel_sizes)) + (lstm_hidden_dim * 2)

        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        embedded = self.embedding(input_ids)
        embedded = self.embedding_dropout(embedded)
        masked_embedded = embedded * mask

        mean_feature = masked_embedded.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        cnn_input = masked_embedded.transpose(1, 2)
        cnn_features = []
        for conv in self.convs:
            conv_output = F.gelu(conv(cnn_input))
            if conv_output.size(-1) > attention_mask.size(1):
                conv_output = conv_output[:, :, :attention_mask.size(1)]
            elif conv_output.size(-1) < attention_mask.size(1):
                pad_size = attention_mask.size(1) - conv_output.size(-1)
                conv_output = F.pad(conv_output, (0, pad_size))
            conv_mask = attention_mask.unsqueeze(1).bool()
            conv_output = conv_output.masked_fill(~conv_mask, -1e4)
            cnn_features.append(conv_output.max(dim=2).values)
        cnn_feature = torch.cat(cnn_features, dim=1)

        lstm_output, _ = self.lstm(masked_embedded)
        attention_scores = self.attention(lstm_output).squeeze(-1)
        attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e4)
        attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)
        attention_feature = (lstm_output * attention_weights).sum(dim=1)

        features = torch.cat([mean_feature, cnn_feature, attention_feature], dim=1)
        logits = self.classifier(features)
        if logits.size(-1) == 1:
            return logits.squeeze(-1)
        return logits

    def get_word_embedding_matrix(self) -> torch.Tensor:
        return self.embedding.weight.detach()

    def get_sentence_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        embedded = self.embedding(input_ids)
        masked_embedded = embedded * mask
        return masked_embedded.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def build_sentiment_model(
    model_type: str,
    vocab_size: int,
    embedding_dim: int,
    hidden_dim: int,
    dropout: float,
    padding_idx: int = 0,
    cnn_num_filters: int = 128,
    cnn_kernel_sizes: str = "3,5,7",
    lstm_hidden_dim: int = 128,
    lstm_layers: int = 2,
    num_classes: int = 1,
) -> nn.Module:
    if model_type == "embedding_dnn":
        return SentimentEmbeddingDNN(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_classes=num_classes,
            padding_idx=padding_idx,
        )

    if model_type == "hybrid_cnn_lstm_attention":
        return HybridCnnLstmAttentionSentimentModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            cnn_num_filters=cnn_num_filters,
            cnn_kernel_sizes=cnn_kernel_sizes,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layers,
            dropout=dropout,
            num_classes=num_classes,
            padding_idx=padding_idx,
        )

    raise ValueError(f"Unknown model_type: {model_type}")
