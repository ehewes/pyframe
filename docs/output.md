# Output reference

Every PyFrame scan returns a `ScanResult`. This page documents its full JSON
shape, field by field.

## Getting the output

CLI (machine-readable):

```bash
pyframe clip.gif --backend local --json
```

Python:

```python
from pyframe import Pipe

result = Pipe("clip.gif", backend="local").run()

result.to_json()   # JSON string
result.to_dict()   # plain dict
result.is_nsfw     # bool  -> the authoritative pass/fail
result.verdict     # Severity -> prints as "clean" / "uncertain" / "nsfw" / "error"
result.max_score   # float 0..1
result.flagged_frames  # list of per-frame results where is_nsfw is True
```

## Example

A clean image scanned with the local backend:

```json
{
  "source": "media/example.jpeg",
  "media_kind": "image",
  "verdict": "clean",
  "is_nsfw": false,
  "max_score": 0.0204,
  "worst_frame": {
    "score": 0.0204,
    "is_nsfw": false,
    "backend": "local",
    "frame_index": 0,
    "timestamp": 0.0,
    "labels": [
      { "name": "sfw", "confidence": 0.9795508384704590 },
      { "name": "nsfw", "confidence": 0.0204491075128317 }
    ]
  },
  "frames": [ { "...": "same shape as worst_frame" } ],
  "backends_used": ["local"],
  "frames_total": 1,
  "frames_screened": 0,
  "frames_classified": 1,
  "cost_usd": 0.0,
  "prescreen_used": false,
  "escalated": null,
  "windows": 0,
  "elapsed_s": 0.656
}
```

## Top-level fields

| Field | Type | Meaning |
|-------|------|---------|
| `source` | string | The input path exactly as passed in. |
| `media_kind` | string | `"image"` or `"animation"` (GIF/video). |
| `verdict` | string | Overall category: `clean`, `uncertain`, `nsfw`, or `error`. See [Verdict values](#verdict-values). |
| `is_nsfw` | bool | The authoritative pass/fail: `true` if any classified frame met the NSFW threshold. Branch on this. |
| `max_score` | float | Highest NSFW score (0..1) across the classified frames, rounded to 4 dp. |
| `worst_frame` | object \| null | The single highest-scoring frame (a [frame object](#frame-object)), or `null` if nothing was classified. |
| `frames` | array | The [frame objects](#frame-object) that were classified. In a short-circuited clean cascade these are the soft-screen frames. |
| `backends_used` | string[] | Which backends produced scores, e.g. `["local"]` or `["local", "aws"]` for a cascade. |
| `frames_total` | int | Total frames decoded from the media (1 for an image). |
| `frames_screened` | int | Frames the cheap soft-screen looked at. `0` in single-pass. |
| `frames_classified` | int | Frames/grids the precise backend classified. In a cascade this equals the number of precise (e.g. AWS) calls made. |
| `cost_usd` | float | Estimated total cost: precise calls plus screen calls, each times that backend's per-image price. `0.0` for local-only. |
| `prescreen_used` | bool | Whether the two-pass cascade was enabled. |
| `escalated` | bool \| null | Cascade only: `true` if it escalated to the precise backend, `false` if it short-circuited clean. `null` for single-pass and images. |
| `windows` | int | Cascade only: how many distinct flagged time-regions the soft-screen found. `0` otherwise. Informational; it does not drive cost. |
| `elapsed_s` | float \| null | Wall-clock seconds for the scan. |

## Frame object

Each entry in `frames` (and `worst_frame`) is one classified frame:

| Field | Type | Meaning |
|-------|------|---------|
| `score` | float | NSFW score 0..1 for this frame, rounded to 4 dp. |
| `is_nsfw` | bool | `true` if `score` met the active threshold (`min_confidence`). |
| `backend` | string | Which backend scored it: `"local"` or `"aws"`. |
| `frame_index` | int | Index of the frame in the decoded sequence. `-1` or a batch index for a merged grid. |
| `timestamp` | float | Seconds into the clip, rounded to 3 dp. `0.0` for images. |
| `labels` | array | Raw [labels](#label-object) the backend returned. |
| `error` | string | Present only if the backend errored on this frame (then `score` is `0` and `is_nsfw` is `false`). |

## Label object

| Field | Type | Meaning |
|-------|------|---------|
| `name` | string | Backend label. Local models use their own set (e.g. `sfw`/`nsfw`); AWS uses moderation names (e.g. `Explicit Nudity`). |
| `confidence` | float | The backend's confidence for that label, 0..1 (full precision). |
| `taxonomy` | string | Optional parent category. AWS only (its `ParentName`); absent for local backends. |

## Verdict values

`verdict` is derived from `max_score` and the thresholds. `is_nsfw` is the
boolean version; `verdict` adds an "uncertain" band:

| Value | Condition |
|-------|-----------|
| `nsfw` | `max_score >= min_confidence` (threshold default: 0.5 local, 0.8 aws) |
| `uncertain` | `uncertain_threshold <= max_score < min_confidence` (default `uncertain_threshold` 0.3) |
| `clean` | `max_score < uncertain_threshold` |
| `error` | every classified frame failed to score |

## Single-pass vs cascade

The same schema is returned either way; these fields change with the mode:

| Field | Single-pass (default) | Cascade (`prescreen=True`) |
|-------|-----------------------|----------------------------|
| `prescreen_used` | `false` | `true` |
| `frames_screened` | `0` | number of soft-screen frames |
| `escalated` | `null` | `true` (hit precise backend) or `false` (short-circuited clean) |
| `windows` | `0` | number of flagged regions found |
| `frames_classified` | up to `max_frames` | up to `max_escalations` (merged grids) |
| `backends_used` | the one backend | `["local"]` if clean, `["local", "aws"]` if escalated |
| `cost_usd` | per-frame precise cost | `0` if short-circuited, else up to `max_escalations` precise calls |

## CLI exit codes

When run as `pyframe ... ` (without `--json` you still get these), the process
exit code encodes the outcome so it slots into shell gates:

| Code | Meaning |
|------|---------|
| `0` | clean |
| `1` | NSFW (subject to `--fail-on`) |
| `2` | bad input (unsupported type, decode error, missing file) |
| `3` | backend not installed (missing optional extra) |

```bash
pyframe upload.gif --backend local || echo "rejected"
```
