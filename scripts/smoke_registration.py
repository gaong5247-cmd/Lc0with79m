"""Verify LC0 exposes the Maia3 backend without loading the heavy model."""
import queue
import subprocess
import sys
import threading

def main(exe):
    proc = subprocess.Popen([exe, "--backend=random", "--weights="],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdin and proc.stdout
    lines = queue.Queue()
    def collect():
        for line in proc.stdout:
            lines.put(line.strip())
    threading.Thread(target=collect, daemon=True).start()
    try:
        proc.stdin.write("uci\n")
        proc.stdin.flush()
        all_lines = []
        while True:
            try:
                line = lines.get(timeout=25)
            except queue.Empty:
                raise RuntimeError(f"Timed out requesting UCI registration; process return code={proc.poll()}, hex={((proc.poll() or 0)&0xffffffff):08X}")
            print(line, flush=True)
            all_lines.append(line)
            if line == "uciok":
                break
            if proc.poll() is not None:
                raise RuntimeError("LC0 exited before UCI handshake")
        if "maia79" not in "\n".join(all_lines):
            raise AssertionError("maia79 not found in LC0 UCI backend choices")
        print("PASS: Maia79 registered as LC0 backend")
    finally:
        if proc.poll() is None:
            proc.stdin.write("quit\n")
            proc.stdin.flush()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

if __name__ == "__main__":
    main(sys.argv[1])
