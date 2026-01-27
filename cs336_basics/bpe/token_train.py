import argparse
import cProfile
import io
import json
import pstats
import time
import tracemalloc
from pathlib import Path
from typing import Iterable

try:
    from cs336_basics.bpe.bpe import bpeTokenizer
except ImportError:
    from cs336_basics.bpe.bpe import bpeTokenizer


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


def encode_token(token_bytes: bytes, encoder: dict[int, str], special_tokens: set[bytes]) -> str:
    if token_bytes in special_tokens:
        return token_bytes.decode("utf-8")
    return "".join(encoder[b] for b in token_bytes)


def write_vocab_json(
    vocab: dict[int, bytes],
    output_path: Path,
    encoder: dict[int, str],
    special_tokens: set[bytes],
) -> None:
    vocab_as_strings = {
        encode_token(token_bytes, encoder, special_tokens): token_id
        for token_id, token_bytes in vocab.items()
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(vocab_as_strings, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def write_merges_txt(
    merges: Iterable[tuple[bytes, bytes]],
    output_path: Path,
    encoder: dict[int, str],
    special_tokens: set[bytes],
) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for left, right in merges:
            left_str = encode_token(left, encoder, special_tokens)
            right_str = encode_token(right, encoder, special_tokens)
            f.write(f"{left_str} {right_str}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer on TinyStories.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/TinyStoriesV2-GPT4-train.txt"),
        help="Path to the TinyStories training file.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="Maximum vocabulary size (including special tokens).",
    )
    parser.add_argument(
        "--special-token",
        type=str,
        default="<|endoftext|>",
        help="Special token to add to the vocabulary.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to write the vocab and merges files.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Number of processes for pre-tokenization (defaults to CPU count).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable cProfile for tokenizer training.",
    )
    return parser.parse_args()


def _train_and_serialize(args: argparse.Namespace) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    tokenizer = bpeTokenizer()
    train_kwargs = {}
    if args.num_processes is not None:
        train_kwargs["num_processes"] = args.num_processes
    vocab, merges = tokenizer.run_train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=[args.special_token],
        **train_kwargs,
    )

    encoder = bytes_to_unicode()
    special_token_bytes = {args.special_token.encode("utf-8")}

    vocab_path = args.output_dir / "tinystories_bpe_vocab.json"
    merges_path = args.output_dir / "tinystories_bpe_merges.txt"
    write_vocab_json(vocab, vocab_path, encoder, special_token_bytes)
    write_merges_txt(merges, merges_path, encoder, special_token_bytes)

    print(f"Wrote vocab to {vocab_path}")
    print(f"Wrote merges to {merges_path}")
    return vocab, merges


def main() -> None:
    args = parse_args()
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tracemalloc.start()
    start_time = time.time()

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        vocab, _ = _train_and_serialize(args)
        profiler.disable()
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream).sort_stats("cumulative")
        stats.print_stats(25)
        print("Profiling (top 25 by cumulative time):")
        print(stats_stream.getvalue())
    else:
        vocab, _ = _train_and_serialize(args)

    elapsed = time.time() - start_time
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    longest_token = max(vocab.values(), key=len)
    longest_token_bytes = len(longest_token)
    print(f"Training wall time: {elapsed / 3600:.2f} hours ({elapsed:.1f} seconds).")
    print(f"Peak traced memory: {peak / (1024 ** 3):.2f} GB.")
    print(f"Longest token length: {longest_token_bytes} bytes.")
    print(f"Longest token bytes: {longest_token!r}")


if __name__ == "__main__":
    main()
