"""Export original pretrained Maia3-79M to LC0-with-79M ONNX.
Requires the official maia3 Python package and trusted checkpoint from its
official Hugging Face repository. No retraining or weight modification.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import torch

from maia3.model_registry import resolve_model_spec, resolve_checkpoint_path
from maia3.models import MAIA3Model


def export(output: Path, *, checkpoint: str | None = None) -> None:
    spec = resolve_model_spec("maia3-79m")
    cfg = SimpleNamespace(**spec.config)
    source = checkpoint or resolve_checkpoint_path(spec)
    print(f"Loading original Maia3 checkpoint: {source}", flush=True)
    state = torch.load(source, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise TypeError("Unexpected Maia checkpoint format")
    state = {key.replace("smolgen", "gab"): val for key, val in state.items()}
    model = MAIA3Model(cfg).cpu().eval()
    # Never silently deploy random/uninitialized parameters.
    model.load_state_dict(state, strict=True)

    tokens = torch.zeros(1, 64, 97, dtype=torch.float32)
    tokens[:, 8:16, 6] = 1.0  # non-empty deterministic test vector
    self_elo = torch.tensor([2600], dtype=torch.int64)
    oppo_elo = torch.tensor([2600], dtype=torch.int64)
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        reference = model(tokens, self_elo, oppo_elo)
        torch.onnx.export(
            model,
            (tokens, self_elo, oppo_elo),
            str(output),
            input_names=["tokens", "self_elos", "oppo_elos"],
            output_names=["policy", "value", "ponder"],
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
            external_data=False,
        )
    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(
        ["policy", "value"],
        {
            "tokens": tokens.numpy(),
            "self_elos": self_elo.numpy(),
            "oppo_elos": oppo_elo.numpy(),
        },
    )
    for name, result, expected in zip(("policy", "value"), actual, reference):
        err = np.max(np.abs(result - expected.detach().numpy()))
        print(f"{name} parity maximum absolute error: {err:.6g}", flush=True)
        if not np.allclose(result, expected.detach().numpy(), atol=1e-3, rtol=1e-3):
            raise RuntimeError(f"ONNX export parity failed: {name}")
    if actual[0].shape != (1, 4352) or actual[1].shape != (1, 3):
        raise RuntimeError("Unexpected ONNX model output shapes")
    print(f"VERIFIED: {output} ({output.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--output", default="maia3-79m.onnx")
    cli.add_argument("--checkpoint", help="Use an already-downloaded official checkpoint")
    args = cli.parse_args()
    export(Path(args.output), checkpoint=args.checkpoint)
