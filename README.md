# LC0 with Maia3-79M

Experimental integration of the pretrained [Maia3-79M](https://huggingface.co/UofTCSSLab/Maia3-79M) chess model into upstream [Leela Chess Zero](https://github.com/LeelaChessZero/lc0).

## Goal

- Fork upstream LC0 rather than LC0-CF.
- Preserve pretrained Maia3-79M weights where possible; no training from scratch.
- Adapt input/history encoding, player/opponent Elo conditioning, geometric attention bias, policy indexing, and WDL outputs.
- Build a Windows x64 UCI engine with GitHub Actions.
- Compare inference against the official Maia3 implementation before claiming compatibility.

## Status

**Planning / integration not yet implemented.** No working LC0-compatible `.pb.gz`, Windows executable, or passing inference tests are provided by this repository yet. Standard LC0 `.pb.gz` weights are not interchangeable with Maia3 ONNX weights.

## Upstream projects

- https://github.com/CSSLab/maia3
- https://huggingface.co/UofTCSSLab/Maia3-79M
- https://github.com/LeelaChessZero/lc0

## Licensing

Maia3 and its weights are distributed under AGPL-3.0; upstream LC0 is GPL-3.0-or-later. Before distributing a combined binary, review license compatibility and provide the appropriate notices, source code, and model attribution. Upstream licenses remain applicable to reused components.

## Initial implementation milestones

1. Pin upstream source and model revisions; audit actual checkpoint tensor names and dimensions.
2. Add reproducible Maia3-vs-adapter test vectors (board state, history, Elo, policy and WDL).
3. Implement an LC0 inference adapter, preferably first via ONNX Runtime, without retraining.
4. Verify legal-move probabilities and WDL against reference Maia3.
5. Add Windows build and UCI smoke tests in GitHub Actions.
6. Evaluate whether a native LC0 protobuf export is feasible; do not assume renaming an ONNX file to `.pb.gz` is sufficient.
