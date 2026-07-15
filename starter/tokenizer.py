"""Byte-level BPE tokenizer (GPT-2 style), trained ONLY on train_corpus.txt
via train_bpe.py, which writes bpe_vocab.json next to this file.

Interface required by train.py / evaluate.py:
    load() -> tokenizer with .encode(str)->list[int], .decode(list[int])->str,
    .vocab_size. Called with NO arguments.
"""
import json
import os

_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_VOCAB_PATH = os.path.join(_DIR, "bpe_vocab.json")


def _char_class(ch):
    if ch.isspace():
        return "S"
    if ch.isalpha():
        return "L"
    if ch.isdigit():
        return "N"
    return "O"


def _get_word_chunks(text):
    """Same pretokenizer used at training time: groups runs of one
    character class, attaching a single leading space to the next word
    (GPT-2 style) so 'word' and ' word' behave consistently. Uses
    str.isalpha()/isdigit() so it's Unicode-aware (works for Devanagari)
    with zero third-party dependencies."""
    chunks = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            if j - i == 1 and j < n:
                k = j
                cls = _char_class(text[j])
                while k < n and _char_class(text[k]) == cls:
                    k += 1
                chunks.append(text[i:k])
                i = k
            else:
                chunks.append(text[i:j])
                i = j
        else:
            cls = _char_class(c)
            j = i + 1
            while j < n and _char_class(text[j]) == cls:
                j += 1
            chunks.append(text[i:j])
            i = j
    return chunks


class BPETokenizer:
    def __init__(self, vocab_size, merges, vocab_bytes):
        self.vocab_size = vocab_size
        # merge order = priority: earlier-learned merges applied first
        self.rank = {pair: i for i, pair in enumerate(merges)}
        self.vocab_bytes = vocab_bytes  # id -> bytes, for O(1) decode
        self.cache = {}  # O(1) cache for fast encoding of chunks

    def _bpe_word(self, symbols):
        """symbols: list[int] (byte values or merged ids). Greedily apply
        the lowest-rank (earliest-learned) applicable merge until none
        remain -- standard BPE encode."""
        if len(symbols) < 2:
            return symbols
        while True:
            best_pair, best_rank = None, None
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                r = self.rank.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, pair
            if best_pair is None:
                return symbols
            a, b = best_pair
            new_id = 256 + best_rank
            merged = []
            i = 0
            while i < len(symbols):
                if (i < len(symbols) - 1 and symbols[i] == a
                        and symbols[i + 1] == b):
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged

    def encode(self, text):
        ids = []
        for chunk in _get_word_chunks(text):
            if chunk not in self.cache:
                symbols = list(chunk.encode("utf-8"))
                self.cache[chunk] = self._bpe_word(symbols)
            ids.extend(self.cache[chunk])
        return ids

    def decode(self, ids):
        b = b"".join(self.vocab_bytes[i] for i in ids)
        return b.decode("utf-8", errors="replace")


class ByteTokenizer:
    """Fallback used only if bpe_vocab.json is missing (e.g. before
    train_bpe.py has been run). Kept so the pipeline never hard-fails."""
    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")


def load(path=None):
    """Return the tokenizer used by train.py / evaluate.py."""
    vocab_path = path or _DEFAULT_VOCAB_PATH
    if not os.path.exists(vocab_path):
        return ByteTokenizer()
    with open(vocab_path, encoding="utf-8") as f:
        data = json.load(f)
    merges = [tuple(p) for p in data["merges"]]
    
    # Reconstruct vocab_bytes dynamically from merges
    vocab_bytes = {i: bytes([i]) for i in range(256)}
    for idx, (a, b) in enumerate(merges):
        new_id = 256 + idx
        vocab_bytes[new_id] = vocab_bytes[a] + vocab_bytes[b]
        
    return BPETokenizer(data["vocab_size"], merges, vocab_bytes)
