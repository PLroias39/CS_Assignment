# tokenizer.py
# include run_bpe and Tokenizer-Class
#

import os
import regex
import collections

# Initialize Vocabulary
def init_vocab_v1_base(special_tokens: list[str]) -> dict[int: bytes]:
    vocab = {}
    # put special_tokens
    for i, st in enumerate(special_tokens):
        vocab[i] = st.encode('utf-8')
    # put base-bytes
    offset = len(special_tokens)
    for i in range(256):
        vocab[offset + i] = bytes([i])
    
    return vocab

def _test_init_vocab(init_fn):
    special_tokens = ["<|endoftext|>", ]
    vocab = init_fn(special_tokens)
    for id, token in vocab.items():
        print(f"{id}:{repr(token)}", end="\t")
        if (id + 1) % 5 == 0:
            print()
        if id == len(vocab):
            print()

# Pre Tokenization
def pre_tokenization_v1_base(input_path: str | os.PathLike, special_tokens: list[str]) -> list[str]:
    # 1. read path
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 2. Split by special_tokens: doc -> sample + <special_tokens> + sample
    special_PAT = regex.compile("(" + "|".join(regex.escape(t) for t in sorted(special_tokens, key=len, reverse=True)) + ")")
    parts = special_PAT.split(content)
    # 3. Split by regex: sample -> strings
    # 可放到函数外进行预编译
    PAT =regex.compile( r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    chunk_tokens = []
    special_set = set(special_tokens)

    for part in parts:
        if not part:
            continue
        if part in special_set:
            continue
        chunk_tokens.extend(PAT.findall(part))
    return chunk_tokens

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
    chunk_tokens = init_fn(test_file, special_tokens)
    for i in range(0, len(chunk_tokens), 5):
        batch = chunk_tokens[i: i+5]
        print("\t".join([repr(t) for t in batch]))
    if os.path.exists(test_file):
        os.remove(test_file)
# Get frequency of words
# and transfer bytes -> int by vocab
def get_word_freqs_v1_base(chunk_tokens: list[str], offset: int) -> dict[tuple[int, ...], int]:
    """ tip: offset cause by special_tokens, base-bytes 's id doesn't equal to ASCII """
    raw_counts = collections.Counter(chunk_tokens)
    word_freqs = {}
    for text_chunk, freq in raw_counts.items():
        # transfer string -> bytes
        # "Hello" -> b'Hello'
        b_chunk = text_chunk.encode('utf-8')
        # transfer bytes -> tuple[int, ...]
        # b'Hello' -> (72, 101, 108 ,108, 111)
        # tip: should add offset
        byte_tuple = tuple(b + offset for b in b_chunk)

        word_freqs[byte_tuple] = freq
    return word_freqs

def _test_get_word_freqs(init_fn):
    chunk_tokens = ["abc", ",", "ab", "abc"]
    offset = 1
    word_freqs = init_fn(chunk_tokens, offset)
    print(f"word_freqs:{word_freqs}")

# Get frequency of pairs
def get_stats_v1_base(word_freqs: dict[tuple[int, ...], int]) -> collections.Counter:
    pairs = collections.Counter()
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pairs[pair] += freq
    return pairs

def _test_get_stats(init_fn):
    word_freqs = {(97, 98, 99): 2, (44,): 1, (97, 98): 1}
    pairs = init_fn(word_freqs)
    print(f"pairs:{pairs}")

# merge in single word
def merge_word_v1_base(word: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i+1]) == pair:
            new_word.append(new_id)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)

def merge2all_v1_base(word_freqs: dict[tuple[int, ...], int], pair: tuple[int, int], new_id:int) -> dict[tuple[int, ...], int]:
    new_word_freqs = {}
    for word, freq in word_freqs.items():
        # 尽可能少的调用merge_word
        if pair[0] in word and pair[1] in word:
            new_word = merge_word_v1_base(word, pair, new_id)
            new_word_freqs[new_word] = new_word_freqs.get(new_word, 0) + freq
        else:
            new_word_freqs[word] = new_word_freqs.get(word, 0) + freq
    return new_word_freqs

def run_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # 1. Initialize Vocabulary
    vocab = init_vocab_v1_base(special_tokens)
    # 2. Pre Tokenization
    chunk_tokens = pre_tokenization_v1_base(input_path, special_tokens)
    # transfer to dict of tuple[int, ...]: int
    # such as {(96, 97, 98):10, }
    offset = len(special_tokens)
    word_freqs = get_word_freqs_v1_base(chunk_tokens, offset)
    
    merge_rules = []
    num_merges = vocab_size - len(vocab)
    current_id = len(vocab)

    for i in range(num_merges):
        # 3. Get frequency
        stats = get_stats_v1_base(word_freqs)
        if not stats:
            break
        # get best_pair: heigher freq / smaller id
        best_pair = max(stats, key=lambda k: (stats[k], -k[0], -k[1]))
        merge_rules.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        # update
        vocab[current_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        # 4. Merge Vocab
        word_freqs = merge2all_v1_base(word_freqs, best_pair, current_id)

        current_id += 1
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

if __name__ == "__main__":
    pass
    # _test_init_vocab(init_vocab_v1_base)
    # _test_pre_tokenization(pre_tokenization_v1_base)
    # _test_get_word_freqs(get_word_freqs_v1_base)
    # _test_get_stats(get_stats_v1_base)
    # _test_main_loop(run_bpe)
