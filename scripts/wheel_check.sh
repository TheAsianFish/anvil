#!/usr/bin/env bash
# Release check: build the wheel, install it into a fresh venv with nothing else from this
# checkout, then run `anvil --help` and the offline demo against that install.
#
# The venv resolves dependencies from PyPI the way `pip install anvil-distil[export,serve]`
# would, not from uv.lock, so a dependency range that only works at the locked versions fails
# here. torch comes from the CPU wheel index (`--torch-backend cpu`) to keep the download small.
# The script runs from a temporary directory so `import anvil` cannot find the source tree.
#
#   scripts/wheel_check.sh [work-dir]     default: a new temporary directory, removed on success
#   WHEEL=dist/anvil_distil-X-py3-none-any.whl scripts/wheel_check.sh   check a wheel already built
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="${1:-$(mktemp -d)}"
mkdir -p "$work"
work="$(cd "$work" && pwd)"
start=$(date +%s)

if [ -n "${WHEEL:-}" ]; then
  wheel="$(cd "$(dirname "$WHEEL")" && pwd)/$(basename "$WHEEL")"
  echo "== checking $(basename "$wheel")"
else
  echo "== build the wheel"
  rm -rf "$work/dist"
  uv build --wheel --out-dir "$work/dist" "$here"
  wheel=$(ls "$work"/dist/anvil_distil-*.whl)
  echo "built $(basename "$wheel")"
fi

echo "== install it into a fresh venv (Python 3.12, CPU torch, dependencies from PyPI)"
rm -rf "$work/venv"
uv venv --quiet --python 3.12 "$work/venv"
uv pip install --quiet --python "$work/venv/bin/python" --torch-backend cpu \
  "${wheel}[export,serve]"
bin="$work/venv/bin"

cd "$work"
where=$("$bin/python" -c 'import anvil, pathlib; print(pathlib.Path(anvil.__file__).parent)')
case "$where" in
  "$work/venv/"*) echo "anvil imports from $where" ;;
  *) echo "anvil imported from $where, not the fresh venv" >&2; exit 1 ;;
esac

echo "== anvil --version / --help"
"$bin/anvil" --version
"$bin/anvil" --help

echo "== the demo against the installed wheel"
ANVIL_CMD="$bin/anvil" PYTHON_CMD="$bin/python" "$here/scripts/demo.sh" "$work/.anvil/demo"

echo "== wheel check passed in $(( $(date +%s) - start ))s"
if [ $# -eq 0 ]; then
  rm -rf "$work"
fi
