"""
Atlas Engine — BPE Tokenizer
Byte-pair encoding tokenizer trained from scratch. Alternatively use tiktoken.
"""
import re, json, os
from collections import defaultdict

class BPETokenizer:
    def __init__(self, vocab_size=8000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}
        self.inv_vocab = {}

    def _get_pairs(self, ids):
        pairs = defaultdict(int)
        for i in range(len(ids) - 1):
            pairs[(ids[i], ids[i+1])] += 1
        return pairs

    def train(self, text):
        tokens = list(text.encode("utf-8"))
        vocab = {i: bytes([i]) for i in range(256)}
        next_id = 256
        ids = tokens[:]
        num_merges = self.vocab_size - 256
        for i in range(num_merges):
            pairs = self._get_pairs(ids)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = next_id + i
            vocab[new_id] = vocab[best[0]] + vocab[best[1]]
            self.merges[best] = new_id
            ids = self._merge(ids, best, new_id)
            if (i+1) % 500 == 0:
                print(f"  merge {i+1}/{num_merges}")
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}

    def _merge(self, ids, pair, new_id):
        out = []
        i = 0
        while i < len(ids):
            if i < len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def _encode_chunk(self, chunk_text, merge_index):
        ids = list(chunk_text.encode("utf-8"))
        while True:
            pairs = self._get_pairs(ids)
            if not pairs:
                break
            mergeable = {p: self.merges[p] for p in pairs if p in self.merges}
            if not mergeable:
                break
            best = min(mergeable, key=lambda p: merge_index.get(p, float('inf')))
            ids = self._merge(ids, best, mergeable[best])
        return ids

    def encode(self, text):
        from concurrent.futures import ThreadPoolExecutor
        chunk_size = 50000
        words = text.split()
        merge_list = list(self.merges.keys())
        merge_index = {p: i for i, p in enumerate(merge_list)}
        chunks = [" ".join(words[c:c+chunk_size])
                  for c in range(0, len(words), chunk_size)]
        total = len(chunks)
        done = [0]
        def encode_one(chunk):
            result = self._encode_chunk(chunk, merge_index)
            done[0] += 1
            if done[0] % 10 == 0 or done[0] == total:
                print(f"  encoded chunk {done[0]}/{total}")
            return result
        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(encode_one, chunks))
        all_ids = []
        for r in results:
            all_ids.extend(r)
        return all_ids

    def decode(self, ids):
        b = b"".join(self.vocab[i] for i in ids)
        return b.decode("utf-8", errors="replace")

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        merges_str = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        vocab_str = {str(k): list(v) for k, v in self.vocab.items()}
        with open(f"{path}/tokenizer.json", "w") as f:
            json.dump({"merges": merges_str, "vocab": vocab_str, "vocab_size": self.vocab_size}, f)
        print(f"Tokenizer saved to {path}/tokenizer.json")

    def load(self, path):
        with open(f"{path}/tokenizer.json") as f:
            data = json.load(f)
        self.merges = {tuple(int(x) for x in k.split(",")): v for k, v in data["merges"].items()}
        self.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = data["vocab_size"]

if __name__ == "__main__":
    tok = BPETokenizer(vocab_size=8000)
    sample = open("data/train.txt").read()[:500000]
    print("Training tokenizer...")
    tok.train(sample)
    tok.save("checkpoints")
    ids = tok.encode("Hello, world!")
    print("Encoded:", ids)
    print("Decoded:", tok.decode(ids))
