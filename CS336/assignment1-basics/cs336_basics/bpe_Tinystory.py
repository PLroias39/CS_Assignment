# bpe_Tinystory.py
# task_02
# train the vocab_file and merges_file for Tokenizer

import time
import json
import pathlib
from cs336_basics.tokenizer import run_train_bpe_v1

current_file_path = pathlib.Path(__file__).resolve()
root_project_path = current_file_path.parent.parent

train_filepath = root_project_path / "data" / "TinyStoriesV2-GPT4-train.txt"
valid_filepath = root_project_path / "data" / "TinyStoriesV2-GPT4-valid.txt"

vocab_filepath = root_project_path / "data" / "Tinystory_vocab.json"
merges_filepath = root_project_path / "data" / "Tinystory_merges.txt"

VOCAB_SIZE = 10000
SPECIAL_TOKENS = ["<|endoftext|>"]
TEMP_SAMPLE_COUNT = 1000

def creat_temporary_dataset(input_path: pathlib.Path, output_path: pathlib.Path, max_samples: int):
    print(f"[INFO] creat_temporary_dataset with {max_samples} samples.")
    sample_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, "r", encoding='utf-8') as f_in, \
        open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            f_out.write(line)
            if "<|endoftext|>" in line:
                sample_count += 1
                if sample_count >= max_samples:
                    break
    print(f"[DONE]Temporay dataset saved to {output_path}\n")


def run_bpe():
    
    temp_train_filepath = root_project_path / "data" / "temp_train_dataset.tmp"
    try: 
        if not train_filepath.exists():
            raise FileNotFoundError(f"original dataset not found at {train_filepath}.")
        creat_temporary_dataset(train_filepath, temp_train_filepath, max_samples=TEMP_SAMPLE_COUNT)

        start_time = time.time()
        vocab, merges = run_train_bpe_v1(
            input_path = temp_train_filepath,
            vocab_size = VOCAB_SIZE,
            special_tokens = SPECIAL_TOKENS
        )
        end_time = time.time()
        print(f"time spent: {end_time - start_time:.2f}s")
    
        # vocab: JSON format. {"idx": "token_str", ...}
        # merges: txt format. token1 token2 \n ...
        save_vocab = {
            str(idx): token_str.decode('latin-1')
            for idx, token_str in vocab.items()
        }
        with open(vocab_filepath, "w", encoding="utf-8") as f:
            json.dump(save_vocab, f, indent=2, ensure_ascii=False)
        with open(merges_filepath, "w", encoding="utf-8") as f:
            for p1, p2 in merges:
                s1 = p1.decode('latin-1')
                s2 = p2.decode('latin-1')
                f.write(f"{s1} {s2}\n")
    except Exception as e:
        print(f"[ERROR] An error during training: {e}\n")
        raise e

    finally:
        if temp_train_filepath.exists():
            print(f"[CLEANUP] Removing temporary file: {temp_train_filepath}.\n")
            temp_train_filepath.unlink()
        else:
            print(f"[CLEANUP] temporary file not found (maybe creation failed).\n")

    

if __name__ == "__main__":
    run_bpe()
