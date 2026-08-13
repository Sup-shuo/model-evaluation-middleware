# Deployment and privacy guide

Public source and private deployment state should be separate.

Install the required environment dependencies with
`python3 -m pip install -r requirements.txt`.

## Recommended layout

```text
public repository/
├── model_evaluation/
├── config/systems/*.yaml       # placeholders only
├── config/models/*.yaml        # public or fictional identities only
└── config/evaluations/*.yaml   # small reusable examples

private deployment directory/
├── systems/production-gpu.yaml
├── systems/vendor-container.yaml
├── evaluations/nightly.yaml
└── records/                     # optional operational reports
```

The CLI accepts catalog IDs or explicit paths, so private files do not need to
live in the Git repository:

```bash
eval-manager run \
  --system-config /etc/model-evaluation/systems/production-gpu.yaml \
  --evaluation-config /etc/model-evaluation/evaluations/nightly.yaml
```

For local development, `config/private/` is ignored by Git.

## Information that should remain private

- personal usernames and home directories;
- SSH aliases, IP addresses, internal hostnames, and jump-host details;
- shared-storage topology and organization-specific mount paths;
- Conda/venv locations that expose personal or internal directory names;
- bearer tokens, API keys, private keys, and secret environment values;
- model inventory, queues, or evaluation records that are not intended to be public;
- `results/`, `cache/`, and `runtime/` operational state.

Generic paths such as `/usr/local/cuda`, `/usr/local/neuware`, `/opt/venvs`,
`/data/models`, and `/var/cache/model-evaluation` are used in the examples.

## Path ownership and portability

Machine-dependent paths belong in the System file, not in Model or Evaluation
files and not in production Python:

| Path | Configuration owner |
|---|---|
| model mount/root | `models.root` |
| dataset/cache root | `paths.cache` |
| result root | `paths.results` (a project-relative value is supported) |
| CUDA/ROCm/Neuware/CANN installation | `profiles.hardware.<id>.runtime.root` |
| backend executable | `profiles.backend.<id>.executable` |
| backend/evaluator Python environment | `profiles.environment.*` |
| evaluator framework checkout | `profiles.evaluator.<id>.root` |

Model files contain logical repository-relative references. Core combines those
references with the selected System model root. The same Model and Evaluation
can therefore be used on hosts with different mount layouts.

If a runtime root is omitted, an adapter may use an explicit standard
environment variable (`CUDA_HOME`, `CUDA_PATH`, `ROCM_PATH`, `NEUWARE_HOME`,
`ASCEND_HOME_PATH`, or `ASCEND_TOOLKIT_HOME`) and tools available on `PATH`.
It does not assume a filesystem installation root in Python code.

The only absolute filesystem literals permitted in production code are kernel
interfaces used for discovery and process ownership (`/proc`, `/sys`, and
`/dev`). HTTP route fragments and package-relative resource paths are not host
deployment paths. `tests/static_contract_check.py` enforces this boundary.

## External endpoints and secrets

Use `secret_ref`/environment indirection supported by the selected backend
profile. Never write a literal bearer token into YAML. The public repository
contains an internal OpenAI-compatible backend adapter, but no real endpoint or
credential example.

## Publishing checklist

```bash
python scripts/check_public_tree.py
python tests/static_contract_check.py
python -m unittest discover -s tests -p 'test_*.py'
```

Also inspect `git diff --cached` before every push. Automated scans are useful,
but they cannot determine whether a harmless-looking model ID or hostname is
confidential to your organization.

Start the public repository from a clean copy without the private tree's `.git`
directory. Removing a secret from the latest revision does not remove it from
Git history. Commit metadata also publishes the configured author name and
email; select the public identity you intend to expose, preferably a GitHub
noreply address, before the first commit:

```bash
git init
git config user.name "<public-name>"
git config user.email "<github-noreply-email>"
git add .
git diff --cached --check
```

Run the privacy checker once more after staging. Do not copy commits from the
private deployment repository merely to preserve history.

## Keeping a production tree unchanged

Do not sanitize a directory while it is running evaluations. Create a separate
public copy, replace deployment assets in that copy, validate it, and publish
only the copy. Updating public documentation must not require touching active
System or Evaluation files.
