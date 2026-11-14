#!/usr/bin/env python3
"""Export and INT8-quantize the Tier-1 zero-shot NLI model.

Run once, by a maintainer, outside the Docker build — not part of the image.
The runtime image needs `onnxruntime` and a `.onnx` file; it has no reason to
carry `torch`, `optimum` or a `transformers` checkout, all of which this script
needs only to *produce* that file. This mirrors how a training pipeline is kept
out of a serving image generally, and specifically avoids repeating the mistake
`docs/02-lld.md` §12.1 already documents once for the T2 prompt: paying a cost
on every container build that only ever needed to be paid once.

Usage::

    pip install "torch>=2.2,<3" "optimum[exporters]>=1.20" onnx
    python scripts/export_zeroshot_model.py --output ./models

Then either:

* point ``AIR_CLASSIFIER__CLASSIFIER__MODEL_PATH``/``TOKENIZER_PATH`` at the
  output directory for a host-native run (matches the existing ``./models``
  dev convention — see ``.env.example``), or
* publish ``model.onnx``/``tokenizer.json`` somewhere fetchable (a release
  asset, a private registry, an internal artifact store — this script does not
  pick one for you) and update the ``MODEL_URL``/``MODEL_SHA256`` build args in
  the Dockerfile to match, the same way ``scripts/fetch_model.py`` already
  pins today's Tier-1 artifact.

The script prints the sha256 of both output files: whichever hosting option is
chosen, that hash is what gets pinned.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

#: Pinned to a commit, not a branch, for the same reason `fetch_model.py` pins
#: one: a moving `main` would silently change the model a build produces while
#: every other input to it stayed the same.
MODEL_REPO = "MoritzLaurer/deberta-v3-base-zeroshot-v1"
#: Verified against https://huggingface.co/api/models/{repo} at the time this
#: was written (2026-08-22) — confirm it is still current before relying on it
#: for a production build; a repo can move on without this file being updated.
MODEL_REVISION = "c2040e8a2599abd2925fdc716cf40e8052d56ec4"

MODEL_FILENAME = "model.onnx"
TOKENIZER_FILENAME = "tokenizer.json"

#: Opset 14+ is what `optimum`'s DeBERTa-v2 exporter targets; onnxruntime has
#: supported it since 1.10, well below the `onnx` extra's pinned floor.
ONNX_OPSET = 14


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def export_and_quantize(output: Path, *, repo: str, revision: str) -> None:
    # Imported here, not at module scope: everything below is conversion-only
    # tooling the runtime image must never need, and importing it eagerly would
    # make `python scripts/export_zeroshot_model.py --help` require a 2GB
    # torch install just to print usage.
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as scratch_name:
        scratch = Path(scratch_name)
        fp32_dir = scratch / "fp32"

        print(f"[export] downloading + exporting {repo}@{revision} to ONNX (fp32)...")
        model = ORTModelForSequenceClassification.from_pretrained(
            repo, revision=revision, export=True
        )
        model.save_pretrained(fp32_dir)

        tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
        tokenizer.save_pretrained(fp32_dir)
        if not (fp32_dir / TOKENIZER_FILENAME).is_file():
            raise RuntimeError(
                f"{repo} did not publish a fast tokenizer ({TOKENIZER_FILENAME}); "
                "the runtime's `tokenizers` crate needs one, a slow/Python "
                "tokenizer will not load."
            )

        print("[export] dynamic INT8 quantization...")
        quantizer = ORTQuantizer.from_pretrained(fp32_dir)
        # Dynamic, not static: static quantization calibrates activation ranges
        # against a representative dataset, which this general-purpose model
        # has none of by definition — the whole point is scoring arbitrary
        # hypotheses sight-unseen. Dynamic quantization only touches weights
        # (computed once, offline) and quantizes activations on the fly per
        # inference, which needs no calibration set and is the standard choice
        # for CPU-served transformer encoders for exactly that reason.
        #
        # per_channel=True is not the default and is not optional here. Measured
        # directly against this checkpoint: per-tensor dynamic quantization (the
        # `avx512_vnni` preset's default, and plain `quantize_dynamic`'s default)
        # collapses the entailment logit gap between a correct and an incorrect
        # hypothesis from >3 points to a few hundredths — the model still runs,
        # still returns a 2-class score, and is silently useless, which is far
        # worse than failing loudly. DeBERTa's disentangled attention projects
        # through weight matrices with wide per-output-channel magnitude ranges;
        # one scale for the whole matrix (per-tensor) is too coarse for them,
        # one scale per output channel (per-channel) is not. Confirmed against
        # both a topic-classification sanity check (travel/cooking/dancing) and
        # the sentiment hypotheses this service actually uses.
        quant_config = AutoQuantizationConfig.avx2(is_static=False, per_channel=True)
        quantizer.quantize(save_dir=scratch / "int8", quantization_config=quant_config)

        quantized = scratch / "int8" / "model_quantized.onnx"
        shutil.copyfile(quantized, output / MODEL_FILENAME)
        shutil.copyfile(fp32_dir / TOKENIZER_FILENAME, output / TOKENIZER_FILENAME)

    for filename in (MODEL_FILENAME, TOKENIZER_FILENAME):
        path = output / filename
        size_mb = path.stat().st_size / (1 << 20)
        print(f"[export] {filename}: {size_mb:.1f} MiB  sha256={_sha256(path)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=Path("./models"))
    parser.add_argument("--repo", default=MODEL_REPO)
    parser.add_argument("--revision", default=MODEL_REVISION)
    args = parser.parse_args(argv)

    try:
        export_and_quantize(args.output, repo=args.repo, revision=args.revision)
    except ImportError as exc:
        print(
            f"[export] missing dependency: {exc}\n"
            '[export] run: pip install "torch>=2.2,<3" "optimum[exporters]>=1.20" onnx',
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
