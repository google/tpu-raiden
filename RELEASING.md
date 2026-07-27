# Releasing tpu_raiden

## Versioning scheme

The base version is single-sourced from `[project].version` in
[pyproject.toml](pyproject.toml) (read by the Bazel repo rule in
[bazel/wheel_version.bzl](bazel/wheel_version.bzl)). A suffix from the
`WHEEL_VERSION_EXTRAS` environment variable is appended at build time:

| Channel | Version format            | Built by                                   | Install |
|---------|---------------------------|--------------------------------------------|---------|
| nightly | `X.Y.Z.devYYYYMMDDHHMMSS` | `Nightly Wheels` workflow (daily 08:00 UTC) | `pip install --pre tpu_raiden_jax --extra-index-url <registry>/simple/` |
| stable  | `X.Y.Z`                   | `Release Wheels` workflow (on `vX.Y.Z` tag) | `pip install tpu_raiden_jax --extra-index-url <registry>/simple/` |

Both channels publish to the same Artifact Registry (default
`https://us-python.pkg.dev/cloud-tpu-inference-test/tpu-raiden/`, overridable
via the `RAIDEN_REGISTRY_URL` repository variable). pip's pre-release rules
keep the channels separate: `.dev` versions are only ever selected with
`--pre`, so a plain `pip install` always resolves to the latest stable
release.

Two wheels are published per version, one per framework:
`tpu_raiden_jax` (bundles `_tpu_raiden_jax.so`, pulls the jax/jaxlib stack)
and `tpu_raiden_torch` (bundles `_tpu_raiden_host.so` /
`_tpu_raiden_torch.so`, no jax deps). Stable releases additionally attach the
wheels and bare `.so` files to a GitHub Release on the tag.

## The torch_tpu ABI pin

`tpu_raiden_torch` is ABI-coupled to `torch_tpu`: both must resolve the same
libtorch symbols at runtime, so raiden is compiled against a specific
torch_tpu checkout and the exact `torch==X.Y.Z+cpu` that torch_tpu's
per-Python requirements lock pins (see [ci/build_wheel_impl.sh](ci/build_wheel_impl.sh)).
torch_tpu publishes nightlies only (`torch_tpu==0.1.1.devYYYYMMDDHHMMSS`), so
every raiden release must record which torch_tpu build it targets:

- [torch_tpu.version](torch_tpu.version) at the repo root pins the torch_tpu
  commit releases are built against (nightlies build against torch_tpu `main`
  instead).
- Every built wheel artifact ships a `torch_tpu_commit.txt` beside it, and the
  GitHub Release notes carry the same pin.
- The matching `torch_tpu` nightly wheel version must be listed in the
  release's CHANGELOG entry so users can install a compatible pair.

## The JAX stack pin

`tpu_raiden_jax` is version-coupled to jax/jaxlib/libtpu, but unlike the
torch side this pin is enforced by pip itself: the exact versions are baked
into the wheel's `Requires-Dist` metadata (from `JAX_REQUIRES` in
[ci/wheel/BUILD.bazel](ci/wheel/BUILD.bazel), kept in lockstep with
[pyproject.toml](pyproject.toml)), so installing the wheel installs the
matching stack. For visibility, both workflows also extract these pins from
the built wheel into a `jax_pins.txt` beside it, and the GitHub Release notes
list them next to the torch_tpu pin. Bumping the JAX stack for a release
means updating `JAX_REQUIRES` and `pyproject.toml` together in the release
PR.

## Cutting a stable release

1. Pick the release candidate commit on `main` — normally the last-known-good
   commit already published in [lkg.version](lkg.version).
2. Open a release PR that:
   - updates [torch_tpu.version](torch_tpu.version) to the validated
     torch_tpu commit,
   - bumps `[project].version` in `pyproject.toml` to `X.Y.Z`,
   - adds the `X.Y.Z` section to [CHANGELOG.md](CHANGELOG.md), including the
     compatible `torch_tpu` nightly version.
3. (Optional) Dry-run: trigger the `Release Wheels` workflow via
   `workflow_dispatch` on the PR branch — it builds stable-versioned wheels as
   workflow artifacts without publishing anything.
4. Merge the PR, then tag and push:

   ```bash
   git tag vX.Y.Z <merge-commit>
   git push origin vX.Y.Z
   ```

   The `Release Wheels` workflow verifies the tag (must equal the pyproject
   version and be reachable from `main`), rebuilds both wheels with an empty
   version suffix, uploads them to the Artifact Registry, and creates the
   GitHub Release.
5. Immediately after the release, bump `[project].version` on `main` to the
   next patch version `X.Y.(Z+1)`. This keeps nightlies
   (`X.Y.(Z+1).devN`) sorting *above* the just-released `X.Y.Z`, so `--pre`
   users keep receiving fresh builds.

## Prerequisites (one-time repo setup)

- `TORCH_TPU_DEPLOY_KEY` secret: read-only deploy key for the private
  `google-pytorch/torch_tpu` repository (the torch wheel build checks it out).
- The GitHub Actions runner service account needs
  `roles/artifactregistry.writer` on the target registry, or set the
  `RAIDEN_REGISTRY_URL` repository variable to a registry it can write to.
