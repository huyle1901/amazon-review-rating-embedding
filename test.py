from src.text_utils import simple_tokenize, build_vocab, encode_text, build_id_to_word

texts = [
    "This movie is good!",
    "This film is terrible.",
    "The acting is very good.",
]

vocab = build_vocab(texts, max_vocab_size=100)

print(vocab)

input_ids, attention_mask = encode_text(
    "This movie is amazing!",
    vocab,
    max_length=8,
)

print(input_ids)
print(attention_mask)

id_to_word = build_id_to_word(vocab)
print(id_to_word)