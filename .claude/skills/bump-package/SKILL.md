---
name: bump-package
description: Review everything committed since the last version bump, decide the semver increment from this repo's standards and bump history, apply it across every version file, refresh the editable install metadata, and return a ready-to-use commit message. Use when asked to "bump the package/version", cut a release, or add a version bump to a PR.
---

# Bump package version

Decide and apply the next version for this package, grounded in what actually changed since the last bump, then hand back a commit message. Do not invent a version; derive it from the diff and the repo's history.

Optional argument: an explicit level (`major` / `minor` / `patch`) or an exact version (`0.4.0`) overrides the decision in step 3. With no argument, decide it.

## 1. Find the current version and the last bump

- Current version: the `version = "X.Y.Z"` line in `pyproject.toml` (the `[project]` table).
- Last bump commit (the commit that last changed that line):
  ```bash
  git log --oneline -G '^version = ' -- pyproject.toml | head -1
  ```
  Fall back to scanning `git log --oneline` for the most recent `Bump version` / `package bump` commit.

## 2. Review what landed since then

```bash
git log <last-bump>..HEAD --oneline
git diff <last-bump>..HEAD --stat
```
Read the substantive diffs, not just the stat. Classify what changed:
- Public API: any symbol added, removed, renamed, or re-signatured among `src/pyframe/__init__.py`'s `__all__` and the modules it exports.
- Default behavior: anything that changes what the library does out of the box (samplers, thresholds, backends, default output shape, exit codes).
- Docs / comments / tests / internal refactor only.

## 3. Decide the increment

This package is `0.x` (alpha), and every historical bump has been a minor `0.x.0` (there is no patch-level precedent). Use semver, adjusted for that convention:

- **Major** (`X+1.0.0`): a breaking change a user must react to. While the project is `0.x` this is normally still shipped as a minor; only go major after `1.0`.
- **Minor** (`0.Y+1.0`): a new backward-compatible feature, or a change to default runtime behavior (for example a sampler or threshold change). This is the common case here.
- **Patch** (`0.Y.Z+1`): docs-only, internal refactor, or a pure bug fix with no behavior or API change.

When in doubt and the change is user-visible, prefer **minor** to match this repo's cadence. State the chosen level and a one-line reason.

## 4. Apply the bump in every location

Update every version string, not just `pyproject.toml`:
- `pyproject.toml` -> `version = "X.Y.Z"`
- `src/pyframe/__init__.py` -> both fallback lines `__version__ = "X.Y.Z"`

Then confirm nothing was missed:
```bash
grep -rn "<old-version>" --include="*.py" --include="*.toml" --include="*.md" . | grep -v "/.venv/"
```

## 5. Refresh the installed metadata

`__version__` resolves from the installed dist metadata first, so an editable install keeps reporting the OLD version until it is reinstalled:
```bash
.venv/bin/python -m pip install -e . --no-deps -q   # use the project venv if one exists
.venv/bin/python -c "import pyframe; print(pyframe.__version__)"   # must print X.Y.Z
.venv/bin/pyframe --version                                       # must print: pyframe X.Y.Z
```
Run the tests to confirm nothing broke: `.venv/bin/python -m pytest -q`.

## 6. Return the commit message

Match the repo's style: subject `Bump version to X.Y.Z`, imperative and short. Add a brief body summarizing what landed since the last bump and why this increment. Do not use em dashes anywhere in the message (user preference).

Example:
```
Bump version to 0.4.0

Recall-floor fix to the default motion sampler now guarantees time coverage,
a user-visible default-behavior change, plus the temporal action segmentation
reference.
```

## Output

Present, in this order:
1. current -> new version, with the increment level and its one-line reason
2. the files changed (and the grep confirming no stray old version strings)
3. the verification result (`pyframe --version` and tests)
4. the commit message in a copy-pasteable block

Do not run `git commit` unless explicitly asked. Leave the bump staged for the user to commit, and offer to commit it if they want.
