# Maia3-79M integration notes

## Current status
The Windows workflow compiles **unmodified upstream LC0 with a random backend**. It does **not** load Maia3-79M and is **not** a playable Maia3 release.

## Design
1. Keep upstream LC0's C++ UCI and search engine.
2. Introduce an ONNX Runtime-backed Maia3 inference adapter, retaining pretrained Maia3-79M weights.
3. Reproduce Maia3 input encoding including 8-position history and Elo conditioning.
4. Convert policy logits into LC0's legal-move policy indices and WDL into LC0's expected perspective.
5. Verify reference output against official Python Maia3 on identical positions before enabling MCTS.
6. Build Windows x64 with the adapter and package its runtime dependencies.

## Non-goals
- Do not rename ONNX files to pb.gz.
- Do not claim a working Maia3 LC0 binary until inference parity tests pass.
- Do not train a new network unless direct adapter feasibility is disproven.

## Upstream
- https://github.com/LeelaChessZero/lc0
- https://github.com/CSSLab/maia3
- https://huggingface.co/UofTCSSLab/Maia3-79M

Check AGPL-3.0 and GPL-3.0-or-later obligations before distributing any combined derivative.
