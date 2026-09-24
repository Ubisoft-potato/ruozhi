"""
Byte-level BPE tokenizer (HuggingFace `tokenizers`), plus the chat format.

nanochat trains its own BPE with a GPT-4 style split pattern; we do the same,
but with the HF tokenizers trainer so there's nothing to compile on Colab.
The split regex keeps runs of CJK characters together (they are \\p{L}), so
BPE can learn multi-character Chinese words, while punctuation breaks chunks.
"""
import os

from tokenizers import Tokenizer, Regex, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = [
    "<|bos|>",             # every document / conversation starts with this
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
]

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RuozhiTokenizer:
    def __init__(self, tok: Tokenizer):
        self.tok = tok
        self.bos_id = tok.token_to_id("<|bos|>")
        self.user_start = tok.token_to_id("<|user_start|>")
        self.user_end = tok.token_to_id("<|user_end|>")
        self.assistant_start = tok.token_to_id("<|assistant_start|>")
        self.assistant_end = tok.token_to_id("<|assistant_end|>")

    # ------------------------------------------------------------------ io
    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size, length=None):
        tok = Tokenizer(models.BPE(byte_fallback=False, unk_token=None, fuse_unk=False))
        tok.normalizer = None
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=Regex(SPLIT_PATTERN), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        tok.decoder = decoders.ByteLevel()
        tok.post_processor = None
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        tok.train_from_iterator(text_iterator, trainer, length=length)
        return cls(tok)

    @classmethod
    def load(cls, directory):
        return cls(Tokenizer.from_file(os.path.join(directory, "tokenizer.json")))

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        self.tok.save(os.path.join(directory, "tokenizer.json"))

    # -------------------------------------------------------------- basics
    @property
    def vocab_size(self):
        return self.tok.get_vocab_size()

    def encode(self, text, prepend_bos=False):
        ids = self.tok.encode(text, add_special_tokens=False).ids
        return [self.bos_id] + ids if prepend_bos else ids

    def encode_batch(self, texts, prepend_bos=False):
        out = self.tok.encode_batch(texts, add_special_tokens=False)
        if prepend_bos:
            return [[self.bos_id] + e.ids for e in out]
        return [e.ids for e in out]

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False)

    def token_bytes(self):
        """
        Number of UTF-8 bytes each token stands for (0 for special tokens), used to
        report loss in bits-per-byte, which is comparable across vocab sizes.
        In byte-level BPE every character of a token string encodes exactly one byte.
        """
        special = set(self.tok.token_to_id(t) for t in SPECIAL_TOKENS)
        out = [0] * self.vocab_size
        for tok_str, idx in self.tok.get_vocab().items():
            out[idx] = 0 if idx in special else len(tok_str)
        return out

    # ---------------------------------------------------------------- chat
    def render_conversation(self, conversation, max_tokens=None):
        """
        conversation = {"messages": [{"role": "user"|"assistant", "content": str}, ...]}
        Returns (ids, mask) where mask=1 marks tokens the model is trained to predict
        (assistant content + <|assistant_end|>), exactly like nanochat's SFT.
        """
        ids, mask = [self.bos_id], [0]
        for i, msg in enumerate(conversation["messages"]):
            role = msg["role"]
            content = self.encode(msg["content"])
            if role == "user":
                assert i % 2 == 0, "user/assistant must alternate, starting with user"
                ids += [self.user_start] + content + [self.user_end]
                mask += [0] * (len(content) + 2)
            elif role == "assistant":
                assert i % 2 == 1, "user/assistant must alternate, starting with user"
                ids += [self.assistant_start] + content + [self.assistant_end]
                mask += [0] + [1] * (len(content) + 1)
            else:
                raise ValueError(f"unknown role {role}")
        if max_tokens is not None:
            ids, mask = ids[:max_tokens], mask[:max_tokens]
        return ids, mask

    def render_for_completion(self, messages):
        """Prompt ids ending in <|assistant_start|>, ready for the model to reply."""
        ids, _ = self.render_conversation({"messages": messages})
        return ids + [self.assistant_start]
