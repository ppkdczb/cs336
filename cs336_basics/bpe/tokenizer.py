from ast import Dict
import json
from math import e
import os
from pyexpat import model
import token
from idna import decode, encode
from numpy import byte, bytes_
import regex as re
import multiprocessing
from pydantic import BaseModel
from typing import List, Optional, Dict
from typing import BinaryIO
import time

from torch import normal

def gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(161, 173))
    bs += list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    return dict(zip(bs, characters))


def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(161, 173))
    bs += list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    return dict(zip(bs, characters))

class tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: Optional[List[tuple[bytes, bytes]]] = None, special_tokens: Optional[list[str]] = None):
        self.vocab = vocab # num to token [int -> bytes]
        self.vocab_token_to_num = {v: k for k, v in vocab.items()} # token to num [bytes -> int]
        self.merges = merges if merges is not None else []
        self.pair_to_rank = {pair: rank for rank, pair in enumerate(self.merges)} # pair to rank [tuple(bytes, bytes) -> int]
        self.special_tokens = special_tokens if special_tokens is not None else []
        #special_tokens 按长度从长到短排序，确保在编码时优先匹配较长的特殊标记
        self.special_tokens.sort(key=len, reverse=True)
        self.split_pattern = re.compile("|".join(map(re.escape, self.special_tokens))) if self.special_tokens else re.compile(r"^$a")  # never match
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.bytes_to_unicode_map = bytes_to_unicode() # int -> str
        self.unicode_to_bytes_map = {v: k for k, v in self.bytes_to_unicode_map.items()} # str -> int
    @classmethod
    def from_files(
        cls,
        vocab_path: str,
        merges_path: str,
        special_tokens: Optional[List[str]] = None,
    ):
        '''
        Class method that constructs and returns a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special tokens.
        '''
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_path) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open(merges_path) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]
        return cls(vocab, merges, special_tokens)
    
 
    def encode_single(self, text: str) -> List[int]:
        s = text.encode('utf-8') # 该词汇的原始bytes流
        #bytes_list = [self.bytes_to_unicode_map[s[i]].encode('utf-8') for i in range(len(s))]  # list of bytes, 不使用 bytes_to_unicode_map
        
        #print(f"Initial bytes list: {bytes_list}") # [240, 159, 153, 131]
        bytes_list = [s[i:i+1] for i in range(len(s))]  # list of bytes, 每个元素是一个字节的bytes对象
        # for merge in self.merges:
        #     i = 0
        #     new_s = []
        #     while i < len(bytes_list) - 1:
        #         if bytes_list[i] == merge[0] and bytes_list[i+1] == merge[1]:
        #             new_s.append(bytes_list[i] + bytes_list[i+1])
        #             i += 2
        #         else:
        #             new_s.append(bytes_list[i])
        #             i += 1
        #     if i < len(bytes_list):
        #         new_s.append(bytes_list[i])
        #     bytes_list = new_s
        # 优化的合并过程，使用 pair_to_rank 来快速找到需要合并的对
        while True:
            pairs = [(bytes_list[i], bytes_list[i+1]) for i in range(len(bytes_list) - 1)]
            pair_ranks = [self.pair_to_rank.get(pair, float('inf')) for pair in pairs]
            if not pair_ranks:
                break
            min_rank = min(pair_ranks)
            if min_rank == float('inf'):
                break
            merge_pair = self.merges[min_rank]
            new_bytes_list = []
            i = 0
            while i < len(bytes_list):
                if i < len(bytes_list) - 1 and (bytes_list[i], bytes_list[i+1]) == merge_pair:
                    new_bytes_list.append(bytes_list[i] + bytes_list[i+1])
                    i += 2
                else:
                    new_bytes_list.append(bytes_list[i])
                    i += 1
            bytes_list = new_bytes_list
        token_ids = []
        #print(f"Bytes list after merges: {bytes_list}")
        for token in bytes_list:
            token_bytes = token
            token_id = self.vocab_token_to_num.get(token_bytes)
            token_ids.append(token_id)
        #print(f"bytes {bytes_list} to token ids: {token_ids}")
        return token_ids
    
    def encode_sentence(self, text: str) -> List[int]:
        tokens = re.finditer(self.PAT, text)
        token_ids = []
        cnt = 0
        for token in tokens:
            cnt += 1
            #print(f"token {cnt}: {token.group(0)}")
            token_id = self.encode_single(token.group(0))
            token_ids.extend(token_id)
        return token_ids
    
    def encode(self, text: str) -> List[int]:
        last_pos = 0
        final_tokens = []
        #print(f"Encoding text: {text}, split_pattern: {self.split_pattern.findall(text)}")
        iter = self.split_pattern.finditer(text)
        for match in iter:
            normal_text = text[last_pos:match.start()]
            #print("normal_text", normal_text)
            if normal_text:
                normal_token_ids = self.encode_sentence(normal_text)
                #print("normal_token_ids", normal_token_ids)
                final_tokens.extend(normal_token_ids)
            special_token = match.group(0)
            special_token_bytes = special_token.encode('utf-8')
            special_token_id = self.vocab_token_to_num.get(special_token_bytes)
            final_tokens.append(special_token_id)
            last_pos = match.end()
        normal_text = text[last_pos:]
        #print("last normal_text", normal_text)
        if normal_text:
            normal_token_ids = self.encode_sentence(normal_text)
            final_tokens.extend(normal_token_ids)
        #print(f"Final token ids: {final_tokens}")
        return final_tokens
        

    def encode_iterable(self, iterable):
        '''
        encode_iterable 的 Docstring
        
        :param self: 说明
        :param iterable: 迭代器
        :return: 迭代器
        '''
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: List[int]) -> str:
        bytes_list = [self.vocab[id] for id in ids] 
        #str_list = [b.decode('utf-8', errors='ignore') for b in bytes_list]
        #str1 = ''.join(str_list)
        #int_list = [self.unicode_to_bytes_map[s] for s in str1]
        #decoded = bytes(int_list)
        decoded = b''.join(bytes_list)
        return decoded.decode('utf-8', errors='ignore')

    