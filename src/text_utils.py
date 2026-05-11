import re
from collections import Counter
from typing import List, Dict, Tuple

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

PAD_ID = 0
UNK_ID = 1

def simple_tokenize(text: str) -> List[str]:
    """
    Convert a raw sentence into a list of tokens.

    Example:
        "This movie is good!" -> ["this", "movie", "is", "good"]
    """
    text = text.lower().strip()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Replace multiple spaces with one space.
    text = re.sub(r"\s+", " ", text)
    if not text:
        return []
    return text.split()

def build_vocab(texts: List[str], max_vocab_size: int = 10000, min_freq: int = 1) -> Dict[str, int]:
    """
    Build a vocabulary from a list of texts.

    Args:
        texts: A list of raw sentences.
        max_vocab_size: Maximum size of the vocabulary (including PAD and UNK).
        min_freq: Minimum frequency for a token to be included in the vocabulary.
    """

    counter = Counter()
    for text in texts:
        tokens = simple_tokenize(text)
        counter.update(tokens)

    vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}

    remaining_slots = max_vocab_size - len(vocab)
    for word, freq in counter.most_common(remaining_slots):
        if freq < min_freq:
            continue
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab

def encode_text(text:str, vocab: Dict[str, int], max_length: int = 64) -> Tuple[List[int], List[int]]:
    """
    Convert a raw sentence into input_ids and attention_mask.

    Example:
        text = "this movie is good"
        max_length = 8

        input_ids:
            [12, 45, 7, 90, 0, 0, 0, 0]

        attention_mask:
            [1, 1, 1, 1, 0, 0, 0, 0]

    Args:
        text:
            Raw input sentence.
        vocab:
            Dictionary mapping token -> id.
        max_length:
            Fixed sequence length.

    Returns:
        input_ids:
            List of token ids with padding.
        attention_mask:
            1 for real tokens, 0 for padding.
    """
    tokens = simple_tokenize(text)
    input_ids = [vocab.get(token, UNK_ID) for token in tokens]

    # Truncate if too long
    input_ids = input_ids[:max_length]

    attention_mask = [1] * len(input_ids)

    # Pad if too short
    padding_length = max_length - len(input_ids)
    if padding_length > 0:
        input_ids += [PAD_ID] * padding_length
        attention_mask += [0] * padding_length
    return input_ids, attention_mask

def build_id_to_word(vocab: Dict[str, int]) -> Dict[int, str]:
    """
    Convert vocab from token -> id to id -> token.
    """
    return {idx: word for word, idx in vocab.items()}






 
