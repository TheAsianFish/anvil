# Releasing Anvil

A release publishes three things: the `anvil-distil` package on PyPI, two container images
on GHCR (`anvil`, the CLI on CPU torch, and `anvil-serve`, ONNX Runtime with no torch), and,
separately and by hand, trained students on the Hugging Face Hub with a model card rendered
from their measured results. Nothing publishes on a push or a pull request.

## One-time setup (the maintainer)

| What | Where | Why |
|---|---|---|
| A PyPI trusted publisher for `anvil-distil`: owner `TheAsianFish`, repository `anvil`, workflow `release.yml`, environment `pypi` | pypi.org, "Your projects" → "Publishing" (for a first release, "Add a new pending publisher") | The `pypi` job uploads with a short-lived OIDC token; no PyPI token is stored anywhere |
| An environment named `pypi` | GitHub repository → Settings → Environments (optionally with a required reviewer, so every upload waits for a click) | The trusted publisher is bound to it |
| Nothing for GHCR | | The `ghcr` job logs in with the workflow's own `GITHUB_TOKEN` and `packages: write`. Images of a private repository are private; make them public under the package's settings when the repository goes public |
| `HF_TOKEN` (a Hugging Face token with write access), in your shell | huggingface.co → Settings → Access Tokens | Only `anvil push`, run locally, reads it; it is not a repository secret |
| `ANTHROPIC_API_KEY` (optional) | GitHub repository → Settings → Secrets and variables → Actions | Runs the `live` CI job; without it the job reports that it skipped |

No other secret is used.

## Cutting a release

1. Pick the version (`0.1.0` is the first) and set it in two places, then refresh the lock:

   ```bash
   sed -i 's/^version = ".*"/version = "0.1.0"/' pyproject.toml
   sed -i 's/^__version__ = ".*"/__version__ = "0.1.0"/' anvil/__init__.py
   uv lock
   ```

2. Run the gates locally on the release commit:

   ```bash
   make check          # format, lint, mypy strict, offline tests
   make wheel-check    # builds the wheel, installs it in a fresh venv, runs anvil --help and the demo
   make docker         # both images build
   ```

   `make wheel-check` resolves dependencies from PyPI, as a user's `pip install` would, not
   from `uv.lock`; it catches a dependency range that only works at the locked versions.

3. Optionally try the release workflow without publishing: Actions → Release → "Run workflow".
   A manual run builds the wheel and sdist and runs the same wheel check; the publish jobs are
   skipped (`if: github.event_name == 'release'`).

4. Merge to `main`, tag and publish a GitHub release:

   ```bash
   git tag v0.1.0 && git push origin v0.1.0
   gh release create v0.1.0 --title "v0.1.0" --notes-file <notes>
   ```

   The workflow fails before building if the tag (without its `v`) differs from
   `anvil.__version__`. Then: `build` (wheel, sdist, clean-venv check, artifact), `pypi`
   (waits for the environment's reviewer, if one is set) and `ghcr` (both images, tagged with
   the version and `latest`).

5. Check what was published:

   ```bash
   python3.12 -m venv /tmp/anvil-check && /tmp/anvil-check/bin/pip install "anvil-distil[export,serve]"
   /tmp/anvil-check/bin/anvil --version
   docker run --rm ghcr.io/theasianfish/anvil:0.1.0 --help
   docker run --rm --entrypoint anvil ghcr.io/theasianfish/anvil-serve:0.1.0 serve --help
   ```

   On Linux, plain `pip install` pulls the CUDA build of torch (several GB). For a CPU-only
   machine, install torch first from `https://download.pytorch.org/whl/cpu`, or use
   `uv pip install --torch-backend cpu`.

## Publishing a student to the Hugging Face Hub

`anvil push` uploads an export (an ONNX bundle for a classifier, the merged checkpoint for a
generator) with a `README.md` model card. Every number in the card is read from the files
next to the student: the `anvil eval` report (parity with the teacher, with intervals), the
calibration and parity gates in the export metadata, and the label cost from the store. A
card for a student that was never evaluated says so instead of showing numbers.

```bash
anvil eval                                   # the report the card's parity table comes from
anvil export --int8                          # the bundle, parity-gated
anvil push .anvil/<task>/exports/<run> --repo <namespace>/<name> --dry-run   # writes README.md, uploads nothing
export HF_TOKEN=...
anvil push .anvil/<task>/exports/<run> --repo <namespace>/<name>             # private by default; --public to publish
```

The students in RESULTS.md live in the eval workspaces: after `make eval-classify` and
`make eval-export`, the Banking77 bundle is under
`.anvil/eval/banking77-<sample>/exports/<run>/`, and `anvil push` finds its eval report and
label store two directories up. Read the dry-run card before pushing: as long as the teacher
is `gold` (no key yet), the card says the student agrees with the dataset's labels, not with
Claude, and a student distilled from Claude labels is the one worth publishing.

Hub uploads are deliberately not a CI job: the students are not in the repository (a
ModernBERT-base bundle is 600 MB and takes about an hour of CPU to train), so a workflow
would have to retrain them on every run.

## If something goes wrong

- **The tag check failed.** Delete the release and the tag (`gh release delete v0.1.0
  --cleanup-tag`), fix the version, start again from step 1.
- **The PyPI upload failed after the build.** Re-run the `pypi` job from the workflow run; the
  `dist` artifact is kept for the run. A version that reached PyPI can never be uploaded
  again, even after deletion: bump the patch version instead.
- **A bad image was pushed.** Push a fixed release; delete the bad tag under the package's
  settings on GitHub.
