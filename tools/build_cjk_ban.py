#!/usr/bin/env python3
"""Build the token-id ban list that makes CJK output impossible for a model.

    .venv/bin/python tools/build_cjk_ban.py --model <checkpoint> \
        --out cjk_ban_tokens.json

Emit the ids as an OpenAI `logit_bias` map (id -> -100) on a request and the
sampler adds that to the logits before sampling, so every listed token is
unreachable.

Why a naive "does this token decode to a CJK character" filter is wrong: this
is byte-level BPE, so a rare character like U+7F4F is assembled from fragment
tokens ('ç½' + 'ı') that decode to no CJK character on their own. Filtering on
decoded text catches the common whole-character tokens and misses roughly 75%
of the rare codepoints -- exactly the ones that leak.

The filter used here is on bytes instead, and relies on UTF-8 being
self-synchronizing: a codepoint cannot be emitted without its leading byte, so
banning every token that carries a CJK lead byte makes the character
unreachable no matter how the fragments are split.

    3-byte lead 0xE3         U+3000-U+3FFF  CJK punctuation, kana
    3-byte lead 0xE4..0xE9   U+4000-U+9FFF  CJK ideographs
    3-byte lead 0xEF         U+F900-U+FFEF  compatibility, fullwidth forms
    4-byte lead 0xF0 + 0xA0..0xAF           CJK extension B/C/D

Deliberately NOT banned, so ordinary output is untouched: 0xC2-0xD3 (accented
Latin), 0xE2 (U+2000-U+2FFF: em dash, bullets, arrows, math), and 0xF0 0x9F
(emoji). The 4-byte extension planes are closed by banning the single token
that ends in a bare 0xF0 -- no token in this vocabulary contains an inline
`F0 A0..AF`, so that one token is the only way to open a 4-byte CJK sequence,
and emoji survive because they come from tokens carrying `F0 9F` inline.
"""

from __future__ import annotations

import argparse
import json
import sys

# 3-byte UTF-8 lead bytes that begin a CJK codepoint.
CJK_LEAD_3B = set(range(0xE3, 0xEA)) | {0xEF}
# Second byte of a 4-byte sequence for the CJK extension planes (U+20000+).
EXT_SECOND = range(0xA0, 0xB0)


def byte_decoder() -> dict[str, int]:
    """Inverse of the GPT-2 byte-level BPE printable-character map."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Build a CJK token ban list for logit_bias."
    )
    p.add_argument("--model", required=True, help="checkpoint dir or HF id")
    p.add_argument("--out", required=True, help="where to write the JSON id list")
    p.add_argument(
        "--bias", type=float, default=-100.0, help="bias value for --emit-map"
    )
    p.add_argument(
        "--emit-map",
        action="store_true",
        help="write {id: bias} ready to paste as logit_bias, instead of a bare id list",
    )
    p.add_argument(
        "--verify", action="store_true", help="prove no Han codepoint escapes"
    )
    args = p.parse_args(argv)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dec = byte_decoder()

    def token_bytes(i: int) -> bytes | None:
        t = tok.convert_ids_to_tokens(i)
        if not isinstance(t, str):
            return None
        try:
            return bytes(dec[c] for c in t)
        except KeyError:
            return None  # special / added token, no byte form

    ban: set[int] = set()
    for i in range(len(tok)):
        b = token_bytes(i)
        if not b:
            continue
        if any(x in CJK_LEAD_3B for x in b):
            ban.add(i)
            continue
        # A bare trailing 0xF0 lets the next token supply 0xA0..0xAF and open
        # a 4-byte CJK extension sequence.
        if b[-1] == 0xF0:
            ban.add(i)
            continue
        if any(b[j] == 0xF0 and b[j + 1] in EXT_SECOND for j in range(len(b) - 1)):
            ban.add(i)

    print(
        f"banned {len(ban)} of {len(tok)} tokens ({100 * len(ban) / len(tok):.1f}%)",
        file=sys.stderr,
    )

    if args.verify:
        ranges = [(0x4E00, 0xA000, 7), (0x3400, 0x4DC0, 13), (0x20000, 0x2A6E0, 17)]
        escapes = tested = 0
        for lo, hi, step in ranges:
            for cp in range(lo, hi, step):
                ids = tok.encode(chr(cp), add_special_tokens=False)
                tested += 1
                if ids and not any(i in ban for i in ids):
                    escapes += 1
        print(
            f"verify: {tested} Han codepoints tested, {escapes} escapes",
            file=sys.stderr,
        )
        for s in ["naïve café Straße", "— → ≤ ∑ •", "🚀🔥", 'JSON {"a":1}']:
            hit = [i for i in tok.encode(s, add_special_tokens=False) if i in ban]
            print(f"  {'OK     ' if not hit else 'BLOCKED'} {s!r}", file=sys.stderr)
        if escapes:
            return 1

    payload = {str(i): args.bias for i in sorted(ban)} if args.emit_map else sorted(ban)
    with open(args.out, "w") as f:
        json.dump(payload, f)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
