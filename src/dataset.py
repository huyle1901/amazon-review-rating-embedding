from typing import Dict, List
import torch
from torch.utils.data import Dataset

from src.text_utils import encode_text

class SentimentDataset(Dataset):
    """
    PyTorch Dataset for sentiment classification.

    Each item returns:
        input_ids: Tensor of token ids
        attention_mask: Tensor marking real tokens vs padding
        label: Tensor label id
    """
    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int], max_length: int = 64):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

        if len(texts) != len(labels):
            raise ValueError("Texts and labels must have the same length.")
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        input_ids, attention_mask = encode_text(text, self.vocab, self.max_length)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }
