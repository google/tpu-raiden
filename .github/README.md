# tpu-raiden CI/CD

GitHub Actions CI for tpu-raiden, modeled on
[`torch_tpu`](https://github.com/google-pytorch/torch_tpu)'s setup but scoped
**CPU-only** for now. The existing Kokoro jobs under [`kokoro/`](../kokoro) are
kept and run alongside these.

## Why CPU-only

torch_tpu runs TPU lanes on self-hosted runner scale-sets
(`linux-x86-ct5lp/ct6e/tpu7x`), but those are granted **per org/repo** and
torch_tpu lives in the `google-pytorch` org — that access does **not**
automatically extend to `google/tpu-raiden`. Until this repo is granted TPU
runner scale-sets, all GitHub Actions CI here stays on a CPU runner
(`linux-x86-n2-32`). TPU/device tests continue to run via Kokoro or on a
developer TPU VM. To add TPU lanes later, grant the repo the scale-set, add the
label to [`actionlint.yaml`](actionlint.yaml), and reintroduce a privileged TPU
job (the `git` history has a v5e/v6e/v7 matrix to crib from).

> The `linux-x86-n2-32` label is what we expect to have; if the available CPU
> scale-set differs, change `runs-on:` in `presubmit.yml` / `nightly.yml` and
> the label in `actionlint.yaml`.

## Workflows

| Workflow | Trigger | What it does |
| --- | --- | --- |
| [`lint.yml`](workflows/lint.yml) | PR | Apache license headers (`addlicense`) + commit-message check. GitHub-hosted. |
| [`clang_format.yml`](workflows/clang_format.yml) | PR | Google C++ style on changed `.cc/.h` via pinned `clang-format==18`. GitHub-hosted. |
| [`presubmit.yml`](workflows/presubmit.yml) | PR | Build the JAX extension + run OSS-safe CPU unit tests + import smoke on `linux-x86-n2-32`. |
| [`nightly.yml`](workflows/nightly.yml) | push to `main` / daily cron 09:00 UTC / dispatch | CPU build + unit tests, build the `tpu_raiden_jax` wheel (twine-checked, uploaded as an artifact), and file a tracking issue on failure. |

Shared logic lives in [`ci/tools/`](../ci/tools): `install_clang18.sh` (the
clang-18 the build needs) and `bazel_test.sh` (bazel binary + dummy torch_tpu
override + flags, matching `build.sh`).

## Scope notes

- **JAX-only build.** The JAX path generates a dummy `torch_tpu` Bazel module
  (same as `build.sh` / `kokoro/.../presubmit.sh`), so no secrets are needed. The
  **Torch** extension/wheel additionally needs a `torch_tpu` checkout + a local
  `torch`, and its meaningful tests need a TPU — out of scope here for now.
- **Unit-test set.** `CPU_TEST_TARGETS` is a curated CPU-safe, OSS-loadable set.
  `//rpc/...` and `//kv_cache/global_registry/...` (and the `kv_cache_store`
  tests that depend on them) load Google-internal gRPC rules and are excluded by
  design — see [`ci/wheel/BUILD.bazel`](../ci/wheel/BUILD.bazel). Expand the list
  as more packages become CPU-runnable.

## One-time setup to turn this on

1. **CPU runner** — grant `google/tpu-raiden` access to the `linux-x86-n2-32`
   self-hosted scale-set (or adjust the label to one it already has).
2. **Bazel remote cache** — bucket `gs://tpu-raiden-bazel-cache` (already used by
   Kokoro); the runner service account needs object read (and write for the
   nightly's read-write cache).
3. **Labels** — create `ci:nightly-failed` for the failure-issue action.
   `GITHUB_TOKEN` is provided automatically.

## Landing & running

`google/tpu-raiden` is a one-way Copybara mirror of an internal google3 repo
(commits carry `PiperOrigin-RevId:`), so these files land **internally** and sync
out — there is no external push/PR path. Once on `main`, the CPU workflows fire on
PRs/pushes automatically, and `nightly.yml` can be kicked manually with
`gh workflow run "CI - Nightly (CPU)"` by anyone with write access.
