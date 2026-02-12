# tokenizer.py
# task_01
# include run_bpe and Tokenizer-Class

import os
import regex
import collections
import json
from typing import Iterable, Iterator

# Initialize Vocabulary
def init_vocab_v1_base(special_tokens: list[str]) -> tuple[dict[int: bytes], int]:
    """creat base ASCII Vocabulary and add special_tokens"""
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256

    for token in special_tokens:
        token_bytes = token.encode('utf-8')
        if token_bytes not in vocab.values():
            vocab[next_id] = token_bytes
            next_id += 1

    return vocab, next_id

def _test_init_vocab(init_fn):
    special_tokens = ["<|endoftext|>", "a"]
    vocab, next_id = init_fn(special_tokens)
    for id, token in vocab.items():
        print(f"{id}:{repr(token)}", end="\t")
        if (id + 1) % 5 == 0:
            print()
        if id == len(vocab):
            print()

# Pre Tokenization
def pre_tokenization_v1_base(input_path: str | os.PathLike, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    """read file, split by special_tokens and regex pattern,and initial word frequencies"""
    
    pre_tokens_cnt = collections.defaultdict(int)
    # 1. read path
    # content: "Hello, felys.<|endoftext|>For test."
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 2. Split by special_tokens: doc -> samples
    # chunks: ["Hello, felys", "For test."]
    if special_tokens:
        special_PAT = "|".join(map(regex.escape, special_tokens))
        chunks = regex.split(special_PAT, content)
    else:
        chunks = [content]

    # 3. Split by regex: sample -> strings
    # words:["It", " 's", "felys", "."]
    # 可放到函数外进行预编译
    PAT =regex.compile( r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    for chunk in chunks:
        if not chunk:
            continue
        words = PAT.findall(chunk)
        for word in words:
            # "Hug" -> (b'H', b'u', b'g')
            word_byte_tuple = tuple(bytes([b]) for b in word.encode('utf-8'))
            pre_tokens_cnt[word_byte_tuple] += 1
            
    return pre_tokens_cnt

def _test_pre_tokenization(init_fn):
    test_file = "test-data.txt"
    special_tokens = ["<|endoftext|>", ]
    test_content = """
    HI, this is the test example in BPE training code.<|endoftext|>
    I just wanna testing my code weither success to work<|endoftext|>
    Happy for 2026CS336.<|endoftext|>
    written in 2026-2-4-12:21
    """

    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)

    pre_tokens_cnt = init_fn(test_file, special_tokens)
    for word_byte_tuple, cnt in pre_tokens_cnt.items():
        print(f"{word_byte_tuple}:{cnt}")

    if os.path.exists(test_file):
        os.remove(test_file)

# Get frequency of pairs
def get_stats_v1_base(pre_tokens_cnt: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    pair_counts = collections.defaultdict(int)
    for token_bytes, cnt in pre_tokens_cnt.items():
        # 遍历相邻对
        for i in range(len(token_bytes) - 1):
            pair = (token_bytes[i], token_bytes[i+1])
            pair_counts[pair] += cnt
    return pair_counts

def _test_get_stats(init_fn):
    pre_tokens_cnt = {(b'b', b'a', b'n', b'a', b'n', b'a'): 1}
    pair_counts = init_fn(pre_tokens_cnt)
    print(f"pairs:{pair_counts}")

# Merge tokens
def merge_tokens_v1_base(pre_tokens_cnt: dict[tuple[bytes, ...], int], pair: tuple[bytes, bytes], new_token: bytes) -> dict[tuple[bytes, ...], int]:
    new_pre_tokens_cnt = {}

    for token_bytes, cnt in pre_tokens_cnt.items():
        new_token_list = []
        i = 0
        while i < len(token_bytes):
            # check if exists pair
            if i < len(token_bytes) - 1 and token_bytes[i]==pair[0] and token_bytes[i+1]==pair[1]:
                new_token_list.append(new_token)
                i += 2
            else:
                new_token_list.append(token_bytes[i])
                i += 1
        new_key = tuple(new_token_list)
        if new_key in new_pre_tokens_cnt:
            new_pre_tokens_cnt[new_key] += cnt
        else:
            new_pre_tokens_cnt[new_key] = cnt
    return new_pre_tokens_cnt

def run_train_bpe_v1(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # 1. Initialize Vocabulary
    vocab, next_id = init_vocab_v1_base(special_tokens)

    # 2. Pre Tokenization
    pre_tokens_cnt = pre_tokenization_v1_base(input_path, special_tokens)
 
    merge_rules = []
    while len(vocab) < vocab_size:
        # 3. Get frequency
        pair_counts = get_stats_v1_base(pre_tokens_cnt)
        if not pair_counts:
            break

        # get best_pair
        max_count = max(pair_counts.values())
        candidates = [k for k, v in pair_counts.items() if v == max_count]
        best_pair = max(candidates)

        # update
        vocab[next_id] = best_pair[0] + best_pair[1]
        next_id += 1
        merge_rules.append(best_pair)
        # 4. Merge Vocab
        pre_tokens_cnt = merge_tokens_v1_base(pre_tokens_cnt, best_pair, best_pair[0] + best_pair[1])

    return vocab, merge_rules

def _test_main_loop(init_fn):
    test_file = "test_corpus.txt"
    corpus_text =(
        """abracadabra! abracadabra! [CLS] ☕ [CLS] ☕"""
    )
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(corpus_text)
    print(f"preview of text: \n {corpus_text[:50]}")

    special_tokens = ["[CLS]", "[SEP]"]
    vocab, merge_rules = init_fn(
        input_path = test_file,
        vocab_size = 265,
        special_tokens = special_tokens
    )
    print(f"output of vocab: \n {vocab}")
    print(f"output of merge_rules: \n {merge_rules}")

    if os.path.exists(test_file):
        os.remove(test_file)


class Tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None,  
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        # reverse vocab
        self.byte_to_int = {v: k for k, v in vocab.items()}

    @classmethod
    def from_files(
        cls, 
        vocab_filepath: str, 
        merges_filepath: str, 
        special_tokens: list[str] | None = None
    ):
        """
        import from files 
        data/Tinystory_vocab.json: {"id": "token_str", ...}
        data/Tinystory_merges.txt: token1 token2\n
        """
        # get vocab
        with open(vocab_filepath, "r", encoding='utf-8') as f:
            raw_vocab = json.load(f)
        vocab = {
            int(idx): token_str.encode('latin-1')
            for idx, token_str in raw_vocab.items()
        }
        
        # get merges
        merges = []
        with open(merges_filepath, "r", encoding='utf-8')as f:
            lines = f.readlines()
            start_idx = 0
            if lines and lines[0].startswith("#"):
                start_idx = 1
            for line in lines[start_idx]:
                line = line.strip()
                if not line: continue
                parts = line.split(' ')
                if len(parts) == 2:
                    p1 = parts[0].encode('latin-1')
                    p2 = parts[1].encode('latin-1')
                    merges.append((p1, p2))
        
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
   
    def _merge_word(self, word_byte_tuple: tuple[bytes, ...]) -> tuple[bytes, ...]:
        """(b"T", b"e", b"s", b"t") -> (b"Tes", b"t")"""
        if len(word_byte_tuple) < 2:
            return word_byte_tuple
        
        current_list = list(word_byte_tuple)
        while True:
            best_pair = None
            min_rank = float('inf')
            
            # create pair and rank
            for i in range(len(current_list)-1):
                pair = (current_list[i], current_list[i+1])
                rank = self.merge_rules.get(pair, float('inf'))
                # check if the rank is lower
                if rank < min_rank:
                    min_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            
            # apply change, and add to new_list
            new_list = []
            i = 0
            while i < len(current_list):
                if i < len(current_list)-1 and (current_list[i], current_list[i+1]) == best_pair
                    new_list.append(current_list[i] + current_list[i+1])
                    i += 2
                else:
                    new_list.append(current_list[i])
                    i += 1
            current_list = new_list
            
            # check if only one word
            if len(current_list) == 1:
                break

        return tuple(current_list)
    def encode(self, text: str) -> list[int]:
        words_to_ids = []
        # 1.split by special_tokens, 
        # but keep them.(use regex's parentheses)
        # chunks: ["Hello, felys!", "<|endoftext|>", "For test."]
        if special_tokens:
            special_PAT = f"({'|'.join(map(regex.escape, self.special_tokens))})"
            chunks = regex.split(special_PAT, text)
        else:
            chunks = [text]

        # 2.deal with general text:
        # 2.1 split by regex, 
        # 2.2 split to basic bytes, 
        # 2.3 apply merges, 
        # 2.4 transfer to ids
        # but jump special_tokens
        # words:["It", " 's", "felys", "."]
        PAT =regex.compile( r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        for chunk in chunks:
            if not chunk:
                continue
            # jump special_tokens
            if chunk in self.special_tokens:
                special_bytes = chunk.encode('utf-8')
                words_to_ids.append(self.byte_to_int[special_bytes])
                continue
            # 2.1 split by regex
            words = PAT.findall(chunk)
            for word in words:
                # 2.2 split to basic bytes
                # "Hug" -> (b"H", b"u", b"g")
                word_byte_tuple = tuple(bytes([b]) for b in word.encode('utf-8'))
                # 2.3 apply merges
                # (b"T", b"e", b"s", b"t") -> (b"Tes", b"t")
                merged_byte_tuple = self._merge_word(word_byte_tuple)
                # 2.4 transfer to ids
                for b in merged_byte_tuple:
                    words_to_ids.append(self.byte_to_int[b])
        return words_to_ids
            
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """ 
        make sure process (one part) of (iterable) a time 
        such as (a yield) of (string fd)
        memory-efficient for large files
        """
        for text_line in iterable:
            line_ids = self.encode(text_line)
            for token_id in line_ids:
                yield token_id
    def decode(self, ids: list[int]) -> str:
        pass
if __name__ == "__main__":
    pass
    # _test_init_vocab(init_vocab_v1_base)
    # _test_pre_tokenization(pre_tokenization_v1_base)
    #  _test_get_stats(get_stats_v1_base)
    # _test_main_loop(run_train_bpe_v1)
