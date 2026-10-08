# LC0 with Maia3-79M

Experimental native ONNX Runtime CPU backend for **upstream Leela Chess Zero**, powered by the original pretrained **Maia3-79M**. No retraining and no LC0-CF fork.

## Implementation

- Native C++20 backend: [integration/maia79_backend.cc](integration/maia79_backend.cc)
- Official checkpoint -> ONNX with torch/ORT output parity checking: [scripts/export_maia79.py](scripts/export_maia79.py)
- Apply backend onto a pinned official LC0 checkout: [scripts/patch_lc0.py](scripts/patch_lc0.py)
- Windows MSVC + UCI search CI: [.github/workflows/windows-maia79.yml](.github/workflows/windows-maia79.yml)
- End-to-end runtime check: [scripts/smoke_uci.py](scripts/smoke_uci.py)
- Windows instructions: [docs/RUN-WINDOWS.md](docs/RUN-WINDOWS.md)

**Build status:** [GitHub Actions](https://github.com/gaong5247-cmd/Lc0with79m/actions/workflows/windows-maia79.yml). A usable build is confirmed **only** when the workflow succeeds and publishes the **LC0-Maia3-79M-Windows-x64** artifact. The preliminary baseline workflow produces only unmodified LC0 and is not the Maia3 engine.

## Design

The backend passes LC0's search-history positions through Maia3's native 8-history, 12-piece-per-square representation, conditions inference on Elo, converts source-destination plus promotion logits into probabilities for legal LC0 moves, and converts Maia's L/D/W logits into the LC0 WDL convention. The model is exported without any weight retraining.

It runs an ordinary LC0 C++ search and UCI frontend. The initial integration uses ONNX Runtime CPU, and is **not** a stock LC0-compatible `.pb.gz` file. Chess960 policy equivalence and numerical parity across a variety of positions need further validation.

## Attribution & licenses

- [Maia3 source](https://github.com/CSSLab/maia3) and [Maia3-79M pretrained model](https://huggingface.co/UofTCSSLab/Maia3-79M): AGPL-3.0.
- [Upstream LC0](https://github.com/LeelaChessZero/lc0): GPL-3.0-or-later.
- [ONNX Runtime](https://github.com/microsoft/onnxruntime): MIT.

Redistributors must comply with applicable copyright and license obligations, including providing corresponding source code and license notices. A source overlay does not remove upstream license requirements.
