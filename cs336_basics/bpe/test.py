from ast import Dict
from heapq import merge
import json
from math import e
import os
from pyexpat import model
from idna import encode
import regex as re
import multiprocessing
from pydantic import BaseModel
from typing import List, Optional, Dict
from typing import BinaryIO
import time

vocab_path = "/home/ppkdczb/study/assignment1-basics/cs336_basics/bpe/tinystories_bpe_vocab.json"
merges_path = "/home/ppkdczb/study/assignment1-basics/cs336_basics/bpe/tinystories_bpe_merges.txt"

with open(vocab_path, 'r', encoding='utf-8') as vf:
    vocab_origin = json.load(vf)
    


# int -> bytes
vocab = {int(k): v.encode('utf-8') for v, k in vocab_origin.items()}


with open(merges_path, 'r', encoding='utf-8') as mf:
    merges_lines = mf.readlines()
merges = []
for line in merges_lines:
    left_str, right_str = line.strip().split(' ')
    merges.append((left_str.encode('utf-8'), right_str.encode('utf-8')))
print(merges[:10])


