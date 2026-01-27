from ast import Dict
import json
from math import e
import os
from pyexpat import model
import token
from idna import encode
from numpy import byte
import regex as re
import multiprocessing
from pydantic import BaseModel
from typing import List, Optional, Dict
from typing import BinaryIO
import time

from torch import normal

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
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.split_pattern = re.compile("|".join(map(re.escape, self.special_tokens))) if self.special_tokens else re.compile(r"^$a")  # never match
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.bytes_to_unicode_map = bytes_to_unicode() # int -> str
        self.unicode_to_bytes_map = {v: k for k, v in self.bytes_to_unicode_map.items()} # str -> int
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_path: str,
        special_tokens: Optional[List[str]] = None,
    ):
        '''
        Class method that constructs and returns a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special tokens.
        '''
        with open(vocab_filepath, 'rb') as vf:
            vocab = json.load(vf)
        vocab = {int(k): v.encode('utf-8') for v, k in vocab.items()}
        merges = []
        with open(merges_path, 'r', encoding='utf-8') as mf:
            merges_lines = mf.readlines()
        for line in merges_lines:
            left_str, right_str = line.strip().split(' ')
            merges.append((left_str.encode('utf-8'), right_str.encode('utf-8')))
        return cls(vocab, merges, special_tokens)
    
 
    def encode_single(self, text: str) -> List[int]:
        s = text.encode('utf-8')
        int_list = list(s)
        new_str_list = [self.bytes_to_unicode_map[b] for b in int_list] # list of str
        bytes_list = [c.encode('utf-8') for c in new_str_list] # list of bytes
        #bytes_list = [s[i:i+1] for i in range(len(s))]  # list of bytes, 不使用 bytes_to_unicode_map
        for merge in self.merges:
            i = 0
            new_s = []
            while i < len(bytes_list) - 1:
                if bytes_list[i] == merge[0] and bytes_list[i+1] == merge[1]:
                    new_s.append(bytes_list[i] + bytes_list[i+1])
                    i += 2
                else:
                    new_s.append(bytes_list[i])
                    i += 1
            if i < len(bytes_list):
                new_s.append(bytes_list[i])
            bytes_list = new_s
        token_ids = []
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
        for match in self.split_pattern.finditer(text):
            normal_text = text[last_pos:match.start()]
            #print("normal_text", normal_text)
            if normal_text:
                normal_token_ids = self.encode_sentence(normal_text)
                final_tokens.extend(normal_token_ids)
            special_token = match.group(0)
            special_token_bytes = special_token.encode('utf-8')
            special_token_id = self.vocab_token_to_num.get(special_token_bytes)
            final_tokens.append(special_token_id)
            last_pos = match.end()
        normal_text = text[last_pos:]
        if normal_text:
            normal_token_ids = self.encode_sentence(normal_text)
            final_tokens.extend(normal_token_ids)
        return final_tokens
        

    def encode_iterable(self, iterable):
        '''
        encode_iterable 的 Docstring
        
        :param self: 说明
        :param iterable: 迭代器
        :return: 迭代器
        '''
        for text in iterable:
            yield self.encode(text)

    def decode(self, ids: List[int]) -> str:
        str_list = []
        for token_int in ids:
            token_bytes = self.vocab.get(token_int)
            token_str = token_bytes.decode('utf-8')
            origin_token_list = [chr(self.unicode_to_bytes_map[c]) for c in token_str]
            origin_token = ''.join(origin_token_list)
            str_list.append(origin_token)
            #str_list.append(token_str)
        return ''.join(str_list)


    