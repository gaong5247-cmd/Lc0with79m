"""Full UCI smoke test: fail if Maia backend cannot load and return a move."""
import argparse
import subprocess
import time


def run(executable: str, model: str, nodes: int, timeout: float) -> None:
    if nodes < 1 or timeout <= 0:
        raise ValueError("nodes and timeout must be positive")
    proc = subprocess.Popen(
        [
            executable,
            "--backend=maia79",
            "--weights=",
            f'--backend-opts=model="{model}",selfelo=2600,oppoelo=2600',
            "--threads=1",
            "--max-prefetch=32",
        ],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert proc.stdin and proc.stdout
    try:
        def send(msg):
            proc.stdin.write(msg + "\n")
            proc.stdin.flush()

        send("uci")
        send("isready")
        send("position startpos")
        send(f"go nodes {nodes}")
        deadline = time.monotonic() + timeout
        seen_uci = seen_ready = False
        seen_move = None
        last_info = ""
        # Windows named pipes are not select()-compatible. A blocking reader
        # thread ensures we can enforce the deadline portably.
        import queue
        import threading
        lines = queue.Queue()
        def consume():
            for ln in proc.stdout:
                lines.put(ln.strip())
        threading.Thread(target=consume, daemon=True).start()
        while time.monotonic() < deadline:
            try:
                line = lines.get(timeout=0.3)
            except queue.Empty:
                if proc.poll() is not None:
                    raise RuntimeError(f"Engine stopped unexpectedly: {proc.returncode}")
                continue
            print(line, flush=True)
            if line.startswith("error ") or line.startswith("fatal "):
                raise RuntimeError(f"LC0 reported an error: {line}")
            if line == "uciok":
                seen_uci = True
            if line == "readyok":
                seen_ready = True
            if line.startswith("info "):
                last_info = line
            if line.startswith("bestmove "):
                seen_move = line.split()[1]
                break
        if not seen_uci or not seen_ready or not seen_move or seen_move == "0000":
            raise RuntimeError(
                f"LC0+Maia did not complete {nodes}-node search. "
                f"exit={proc.poll()}, last_info={last_info!r}"
            )
        print(f"PASS: UCI handshake and Maia-backed go nodes {nodes}: {seen_move}", flush=True)
    finally:
        if proc.poll() is None:
            try:
                send("quit")
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
                proc.wait(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=90)
    args = parser.parse_args()
    run(args.exe, args.model, args.nodes, args.timeout)
