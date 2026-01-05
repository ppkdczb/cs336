from ast import Dict
import os
from pyexpat import model
from idna import encode
import regex as re
import multiprocessing
from pydantic import BaseModel

from .pretokenization_example import find_chunk_boundaries

class TokenizerConfig(BaseModel):
    vocab: list[bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str]


def _count_pretokens_range(args: tuple[str, int, int, str, str]) -> dict[tuple[bytes], int]:
    input_path, start, end, pat, split_pattern = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)
    text = chunk.decode("utf-8", errors="ignore")
    if split_pattern:
        segments = re.split(split_pattern, text)
    else:
        segments = [text]
    counts: dict[tuple[bytes], int] = {}
    for segment in segments:
        for match in re.finditer(pat, segment):
            token = match.group(0).encode("utf-8")
            tuples = tuple(token[i : i + 1] for i in range(len(token)))
            counts[tuples] = counts.get(tuples, 0) + 1
    return counts

class bpeTokenizer:

    def __init__(self):
        self.counts = {}  # dict[tuple[bytes], int]
        self.pair_counts = {}  # dict[tuple[bytes, bytes], int]
        self.parts_to_words = {}  # dict[tuple[bytes], list[tuple[bytes]]]
        self.merges = []  # list[tuple[bytes, bytes]]
        self.words_parent = {}  # dict[tuple[bytes], tuple[bytes]]
        self.vocab = {}  # dict[int, bytes]
        self.special_tokens = []  # list[str]
    # print(merge_bytes(b'he', b'llo'))
    
    def merge_bytes(self, a: bytes, b: bytes) -> bytes:
        return a + b

    def find_root(self, word: tuple[bytes]) -> tuple[bytes]:
        path = []
        while self.words_parent[word] != word:
            path.append(word)
            word = self.words_parent[word]
        for p in path:
            self.words_parent[p] = word
        return word

    def run_train_bpe(
        self,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """Given the path to an input corpus, run train a BPE tokenizer and
        output its vocabulary and merges.

        Args:
            input_path (str | os.PathLike): Path to BPE tokenizer training data.
            vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
            special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
                These strings will never be split into multiple tokens, and will always be
                kept as a single token. If these special tokens occur in the `input_path`,
                they are treated as any other string.

        Returns:
            tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
                vocab:
                    The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                    to bytes (token bytes)
                merges:
                    BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                    representing that <token1> was merged with <token2>.
                    Merges are ordered by order of creation.
        """

        # input_path: ../data/TinyStoriesV2-GPT4-valid.txt
        # input_path = "../data/TinyStoriesV2-GPT4-valid.txt"
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # special_tokens = ["<|endoftext|>"]
        # vocab_size = 1000
        n_special = len(special_tokens)
        import regex as re

        for i in range(256):
            self.vocab[i] = bytes([i])
        for token in special_tokens:
            self.vocab[len(self.vocab)] = token.encode("utf-8")

        split_pattern = "|".join(map(re.escape, special_tokens)) if special_tokens else ""
        num_processes = kwargs.get("num_processes", multiprocessing.cpu_count())
        num_processes = max(1, int(num_processes))
        with open(input_path, "rb") as f:
            if special_tokens:
                boundaries = find_chunk_boundaries(
                    f, num_processes, special_tokens[0].encode("utf-8")
                )
            else:
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                boundaries = [0, file_size]
            ranges = list(zip(boundaries[:-1], boundaries[1:]))

        if len(ranges) == 1 or num_processes == 1:
            start, end = ranges[0]
            chunk_counts = _count_pretokens_range(
                (str(input_path), start, end, PAT, split_pattern)
            )
            for token, count in chunk_counts.items():
                self.counts[token] = self.counts.get(token, 0) + count
        else:
            args = [
                (str(input_path), start, end, PAT, split_pattern)
                for start, end in ranges
            ]
            with multiprocessing.Pool(processes=min(num_processes, len(args))) as pool:
                partial_counts = pool.map(_count_pretokens_range, args)
            for chunk_counts in partial_counts:
                for token, count in chunk_counts.items():
                    self.counts[token] = self.counts.get(token, 0) + count

        # initialize words_parent
        for word in self.counts.keys():
            self.words_parent[word] = word
        # print(len(words_parent))

        for key, value in self.counts.items():
            parts = list(key)
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                assert type(pair) == tuple and all(isinstance(p, bytes) for p in pair)
                self.pair_counts[pair] = self.pair_counts.get(pair, 0) + value
                if pair not in self.parts_to_words:
                    self.parts_to_words[pair] = []
                if key not in self.parts_to_words[pair]:
                    self.parts_to_words[pair].append(key)
        round = vocab_size - 256 - n_special
        for _ in range(round):
            print(f"Round {_+1}/{round}")
            pair_max = max(self.pair_counts.items(), key=lambda x: (x[1], x[0]))
            self.merges.append(pair_max[0])
            new_bytes = self.merge_bytes(pair_max[0][0], pair_max[0][1])
            self.vocab[len(self.vocab)] = new_bytes
            seen_roots: set[tuple[bytes, ...]] = set()
            for key in self.parts_to_words[pair_max[0]]:
                # key is tuple[bytes]
                key = self.find_root(key)  # find root of key, then update
                if key in seen_roots:
                    continue
                seen_roots.add(key)
                parts = list(key)
                new_parts = []
                i = 0
                flag = False
                #防止出现已经合并了的情况还进行更新
                while i < len(parts) - 1:
                    if (parts[i], parts[i + 1]) == pair_max[0]:
                        flag = True
                        break
                    i += 1
                if not flag:
                    continue
                #去除旧串的计数
                i = 0
                while i < len(parts) - 1:
                    self.pair_counts[(parts[i], parts[i + 1])] -= self.counts[key]
                    i += 1
                i = 0
                #合并成新的byte串
                while i < len(parts):
                    if i < len(parts) - 1 and (parts[i], parts[i + 1]) == pair_max[0]:
                        new_parts.append(new_bytes)
                        i += 2
                    else:
                        new_parts.append(parts[i])
                        i += 1
                i = 0
                #进行新串的计数
                while i < len(new_parts):
                    if i < len(new_parts) - 1:
                        pair = (new_parts[i], new_parts[i + 1])
                        self.pair_counts[pair] = (
                            self.pair_counts.get(pair, 0) + self.counts[key]
                        )
                    if new_parts[i] == new_bytes:
                        if i + 1 < len(new_parts):
                            pair_next = (new_bytes, new_parts[i + 1])
                            self.parts_to_words.setdefault(pair_next, []).append(
                                tuple(new_parts)
                            )
                        if i > 0:
                            pair_prev = (new_parts[i - 1], new_bytes)
                            self.parts_to_words.setdefault(pair_prev, []).append(
                                tuple(new_parts)
                            )
                    i += 1
                new_key = tuple(new_parts)
                self.words_parent[key] = new_key
                self.words_parent[new_key] = new_key
                self.parts_to_words.setdefault((new_bytes,), []).append(new_key)
                self.counts[new_key] = self.counts.get(new_key, 0) + self.counts[key]
                del self.counts[key]
        return self.vocab, self.merges
