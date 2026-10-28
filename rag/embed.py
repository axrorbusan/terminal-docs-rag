"""Sentence embeddings with all-MiniLM-L6-v2 running on onnxruntime (CPU, no PyTorch).

Same result as sentence-transformers' all-MiniLM-L6-v2: mean pooling over tokens,
then L2 normalisation, so a dot product is the cosine similarity.
"""
from functools import lru_cache

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from . import config

MAX_TOKENS = 256
DIM = 384


class Embedder:
    def __init__(self, model_dir=config.MODEL_DIR):
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tok.enable_truncation(max_length=MAX_TOKENS)
        self.tok.enable_padding(pad_id=0, pad_token="[PAD]")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.sess = ort.InferenceSession(
            str(model_dir / "model.onnx"), opts, providers=["CPUExecutionProvider"]
        )

    def encode(self, texts, batch_size=32):
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok.encode_batch(texts[i : i + batch_size])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            hidden = self.sess.run(
                None,
                {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)},
            )[0]
            m = mask[..., None].astype(np.float32)
            vec = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            vec /= np.linalg.norm(vec, axis=1, keepdims=True)
            out.append(vec.astype(np.float32))
        return np.vstack(out) if out else np.zeros((0, DIM), dtype=np.float32)

    def n_tokens(self, text):
        self.tok.no_truncation()
        n = len(self.tok.encode(text).ids)
        self.tok.enable_truncation(max_length=MAX_TOKENS)
        return n


@lru_cache(maxsize=1)
def get_embedder():
    return Embedder()
