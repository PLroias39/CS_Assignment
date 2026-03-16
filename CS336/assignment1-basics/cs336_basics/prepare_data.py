# prepare_data.py
# task_02
#
# 1. training tokenizer data
# Input: train.txt
# Output: vocab.json, merges.txt
#
# 2. encoding training data
# Input: train.txt, valid.txt, Tokenizer(vocab.json, merges.txt)
# Output: train.bin, valid.bin

import os 
import time
import json
import pathlib
import numpy as np 
from tqdm import tqdm
from cs336_basics.tokenization import run_train_bpe_v1, Tokenizer

# Assets
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
TRAIN_TXT_PATH = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
VALID_TXT_PATH = DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"

# Output file
VOCAB_PATH = DATA_DIR / "Tinystory_vocab.json"
MERGE_PATH = DATA_DIR / "Tinystory_merges.txt"
TRAIN_BIN = DATA_DIR / "TinyStoriesV2-GPT4-train.bin"
VALID_BIN = DATA_DIR / "TinyStoriesV2-GPT4-valid.bin"

# parameters
VOCAB_SIZE = 10000          # target vocab_size, include special_tokens
SPECIAL_TOKENS = ["<|endoftext|>"]
TEMP_SAMPLE_COUNT = 10000    # for resources-limited, just train few of samples 

def creat_temporary_dataset(input_path: pathlib.Path, output_path: pathlib.Path, max_samples: int):
    """
    Create a smaller temporary dataset from source file.
    Read by line and counts by "<|endoftext|>" token 
    Args:
        input_path (Path): Path to source text file
        output_path (Path): Path to temp text file
        max_samples (int): Number of samples
    """
    print(f"\n[Info] creat_temporary_dataset with {max_samples} samples.")
    sample_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, "r", encoding='utf-8') as f_in, \
        open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            f_out.write(line)
            if "<|endoftext|>" in line: # check for counts
                sample_count += 1
                if sample_count >= max_samples:
                    break
    print(f"[Done] Temporay dataset saved to {output_path}\n")


def train_tokenizer(train_path, vocab_path, merge_path):
    print(f"\n[Info] Training BPE Tokenizer (on {TRAIN_TXT_PATH})...")
    try: 
        if not train_path.exists():
            raise FileNotFoundError(f"original dataset not found at {train_path}.")
        # 1. Prepare data 
        temp_train_filepath = DATA_DIR / "temp_train_dataset.tmp"
        creat_temporary_dataset(train_path, temp_train_filepath, max_samples=TEMP_SAMPLE_COUNT)
        
        """
        2. Train BPE 
        return: 
           vocab: Dict[int, bytes] -> {0:b'<|endoftext|>', ...}
           merges: List[Tuple[bytes, bytes]] -> [(b'e', b'd'), ...]
        """
        vocab, merges = run_train_bpe_v1(
            input_path = temp_train_filepath,
            vocab_size = VOCAB_SIZE,
            special_tokens = SPECIAL_TOKENS
        )

        """
        3. Serializes & Save 
        return: 
           vocab: JSON -> {"idx": "token_str", ...}
           merges: txt -> token1 token2 \n ...
        """
        save_vocab = {
            str(idx): token_str.decode('latin-1')   # latin-1 for raw bytes from bpe training 
            for idx, token_str in vocab.items()
        }
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(save_vocab, f, indent=2, ensure_ascii=False)
        with open(merge_path, "w", encoding="utf-8") as f:
            for p1, p2 in merges:
                s1 = p1.decode('latin-1')
                s2 = p2.decode('latin-1')
                f.write(f"{s1} {s2}\n")

    except Exception as e:
        print(f"\n[Error] An error during training: {e}")
        raise e

    finally:
        if temp_train_filepath.exists():
            print(f"\n[Cleanup] Removing temporary file: {temp_train_filepath}.")
            temp_train_filepath.unlink()
        else:
            print(f"\n[Cleanup] temporary file not found (maybe creation failed).")


def tokenize_to_bin(txt_path, bin_path, tokenizer):
    total_size = os.path.getsize(txt_path)

    if bin_path.exists(): bin_path.unlink()
    print(f"\n[Info] Encoding {txt_path} to {bin_path}...")

    with open(txt_path, 'r', encoding='utf-8') as f_in,\
        open(bin_path, 'ab') as f_out, \
        tqdm(
            total = total_size,
            unit = 'B',
            unit_scale = True,
            desc = f"Encoding {txt_path.name}"
        )as pbar:

        chunk_ids = []
        for line in f_in:
            chunk_ids.extend(tokenizer.encode(line))
            pbar.update(len(line.encode('utf-8')))
            # write 
            if len(chunk_ids) > 1_000_000:
                np.array(chunk_ids, dtype=np.uint16).tofile(f_out)
                chunk_ids = []
    
        if chunk_ids:
            np.array(chunk_ids, dtype=np.uint16).tofile(f_out)

    print(f"[Done] Saved at: {bin_path}")


# Trick 1: Overfit a single batch
def create_tiny_dataset(source_bin, dtype=np.uint16, num_tokens=2048):
    """
    Extract dataset(.bin) to debug 
    """
    tiny_bin_path = source_bin.parent / f"tiny_{source_bin.name}"
    if tiny_bin_path.exists():
        return

    print(f"\n[Info] create tiny_bin_path ...")
    data = np.memmap(source_bin, dtype=dtype, mode='r')
    tiny_data = data[:num_tokens]
    tiny_data.tofile(tiny_bin_path)
    print(f"[Done] tiny_data create at {tiny_bin_path} with {num_tokens} tokens")


def main():
    # train_tokenizer(TRAIN_TXT_PATH, VOCAB_PATH, MERGE_PATH)
    tokenizer = Tokenizer.from_files(
        VOCAB_PATH, 
        MERGE_PATH, 
        SPECIAL_TOKENS
    )
    tokenize_to_bin(TRAIN_TXT_PATH, TRAIN_BIN, tokenizer)
    tokenize_to_bin(VALID_TXT_PATH, VALID_BIN, tokenizer)
    create_tiny_dataset(TRAIN_BIN)
    print("\n[Done] All data prepared for training!")

if __name__ == "__main__":
    main()
