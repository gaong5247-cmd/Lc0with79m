"""Apply the Maia3 native backend overlay to the pinned original LC0 tree.

The upstream checkout is a reproducible source input rather than vendored files.
Do not remove or replace upstream copyright and licensing information.
"""
import argparse
from pathlib import Path
import shutil

START = "# BEGIN lc0with79m native Maia3 backend"
END = "# END lc0with79m native Maia3 backend"

MESON_FRAGMENT = """\
# BEGIN lc0with79m native Maia3 backend
maia_ort_dep = cc.find_library('onnxruntime',
  dirs: get_option('onnx_libdir'), required: true)
maia_ort_include = include_directories(get_option('onnx_include'), is_system: true)
deps += maia_ort_dep
includes += maia_ort_include
files += 'src/neural/backends/maia79_backend.cc'
# END lc0with79m native Maia3 backend
"""


def patch(repository: Path, lc0: Path):
    source = repository / "integration" / "maia79_backend.cc"
    target = lc0 / "src" / "neural" / "backends" / "maia79_backend.cc"
    if not (lc0 / "meson.build").exists():
        raise FileNotFoundError(f"LC0 meson.build not found in {lc0}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    meson = lc0 / "meson.build"
    original = meson.read_text(encoding="utf-8")
    if START in original:
        before, after = original.split(START, 1)
        _, after = after.split(END, 1)
        original = before + after.lstrip("\n")
    marker = "#############################################################################\n## Main files"
    if marker not in original:
        raise RuntimeError("Unsupported upstream LC0 meson.build layout")
    original = original.replace(marker, MESON_FRAGMENT + "\n" + marker, 1)
    meson.write_text(original, encoding="utf-8")
    print(f"Patched LC0 source tree: {lc0}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lc0", type=Path, default=Path("upstream/lc0"))
    args = parser.parse_args()
    patch(Path(__file__).resolve().parent.parent, args.lc0)
