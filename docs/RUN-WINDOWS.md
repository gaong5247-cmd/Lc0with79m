# Windows Maia3-79M UCI engine

This package is intended to contain:
- lc0-maia79.exe: upstream LC0 search engine with a Maia3 ONNX inference backend
- maia3-79m.onnx: original pretrained Maia3-79M exported to ONNX
- onnxruntime.dll: ONNX Runtime CPU
- license notices

The source workflow builds these together and must pass a UCI bestmove smoke test before uploading the artifact. An unverified workflow run does not constitute a working build.

## GUI usage
Select lc0-maia79.exe as a UCI chess engine. Set the backend to maia79 if necessary. Elo conditioning defaults to both players at 2600. The model is conditioned to mimic that playing population, not rated 2600 by match results.

## Command line (PowerShell)
From the extracted folder:

    .\lc0-maia79.exe --backend=maia79 --weights= --backend-opts="model=maia3-79m.onnx,selfelo=2600,oppoelo=2600"

Then enter:

    uci
    isready
    position startpos
    go nodes 1
    quit

For a different rating, set selfelo and oppoelo to your choice from 0 to 5000.

## Licensing
This combines Maia3 AGPL-3.0 and LC0 GPL-3.0-or-later. Preserve code/model attribution and the applicable license and source distribution obligations.

## Scope
ONNX Runtime CPU on Windows x64. No DirectML/CUDA support or standard LC0 .pb.gz is claimed.
