from pathlib import Path
import time

import numpy as np

from cs336_basics.bpe.tokenizer import tokenizer


TRAIN_TEXT_PATH = Path("data/TinyStoriesV2-GPT4-train.txt")
VALID_TEXT_PATH = Path("data/TinyStoriesV2-GPT4-valid.txt")
TRAIN_BIN_PATH = Path("data/tinystories_train.bin")
VALID_BIN_PATH = Path("data/tinystories_valid.bin")
VOCAB_PATH = Path("cs336_basics/bpe/tinystories_bpe_vocab.json")
MERGES_PATH = Path("cs336_basics/bpe/tinystories_bpe_merges.txt")
SPECIAL_TOKENS = ["<|endoftext|>"]
OUTPUT_DTYPE = np.uint16
FLUSH_EVERY_TOKENS = 1_000_000


def write_tokenized_corpus(
    tok: tokenizer,
    input_path: Path,
    output_path: Path,
    flush_every_tokens: int = FLUSH_EVERY_TOKENS,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_input_bytes = input_path.stat().st_size
    bytes_read = 0
    total_tokens = 0
    token_buffer: list[int] = []
    start_time = time.perf_counter()

    print(f"Tokenizing {input_path} -> {output_path}")
    with input_path.open("r", encoding="utf-8") as in_f, output_path.open("wb") as out_f:
        for line_idx, line in enumerate(in_f, start=1):
            bytes_read += len(line.encode("utf-8"))
            token_buffer.extend(tok.encode(line))

            if len(token_buffer) >= flush_every_tokens:
                chunk = np.asarray(token_buffer, dtype=OUTPUT_DTYPE)
                chunk.tofile(out_f)
                total_tokens += len(token_buffer)
                progress = bytes_read / total_input_bytes * 100
                elapsed = time.perf_counter() - start_time
                input_mb_per_s = bytes_read / max(elapsed, 1e-9) / 1024**2
                tokens_per_s = total_tokens / max(elapsed, 1e-9)
                print(
                    f"  line={line_idx} progress={progress:.2f}% "
                    f"tokens_written={total_tokens} "
                    f"input_mb_per_s={input_mb_per_s:.2f} "
                    f"tokens_per_s={tokens_per_s:.0f}"
                )
                token_buffer.clear()

        if token_buffer:
            chunk = np.asarray(token_buffer, dtype=OUTPUT_DTYPE)
            chunk.tofile(out_f)
            total_tokens += len(token_buffer)

    total_elapsed = time.perf_counter() - start_time
    avg_input_mb_per_s = bytes_read / max(total_elapsed, 1e-9) / 1024**2
    avg_tokens_per_s = total_tokens / max(total_elapsed, 1e-9)
    print(
        f"Finished {input_path.name}: wrote {total_tokens} tokens, "
        f"avg_input_mb_per_s={avg_input_mb_per_s:.2f}, "
        f"avg_tokens_per_s={avg_tokens_per_s:.0f}, "
        f"elapsed={total_elapsed:.2f}s"
    )


def main() -> None:
    tok = tokenizer.from_files(
        vocab_path=str(VOCAB_PATH),
        merges_path=str(MERGES_PATH),
        special_tokens=SPECIAL_TOKENS,
    )
    write_tokenized_corpus(tok, TRAIN_TEXT_PATH, TRAIN_BIN_PATH)
    write_tokenized_corpus(tok, VALID_TEXT_PATH, VALID_BIN_PATH)


if __name__ == "__main__":
    main()
