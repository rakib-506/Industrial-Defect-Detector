---
title: Industrial Defect Detector
emoji: 🔍
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 5.50.0
app_file: deploy/gradio_app.py
pinned: false
license: cc-by-nc-sa-4.0
short_description: PatchCore anomaly detection with defect-type explainability
---

# Industrial Defect Detector

Upload a photo of a manufactured part. The detector decides whether it is faulty,
shows where, and names the defect type.

It learns from **defect-free examples only**, so it flags failure modes nobody
labelled — the thing a classifier trained on known defects cannot do.

Seven products are trained: `bottle`, `capsule`, `metal_nut` (rigid objects) and
`carpet`, `grid`, `tile`, `wood` (repeating textures).

## Using the demo

1. Pick the product that matches your photo. **This matters** — the result is
   meaningless against the wrong product. A badly mismatched photo is usually
   caught and returned as `UNCERTAIN`, but similar-looking products are not.
2. Upload an image.
3. Read the verdict, the heatmap, and the defect type.

## API

One authenticated endpoint, `predict`, returning the same JSON as the project's
FastAPI backend, plus `metadata` for the category list.

```python
from gradio_client import Client, handle_file

client = Client("YOUR-USERNAME/defect-detector")
result = client.predict(
    image=handle_file("part.png"),
    category="bottle",
    api_key="<the secret>",
    api_name="/predict",
)
print(result["verdict"], result["defect_type"])
```

Over plain HTTP it is Gradio's two-step protocol: `POST /gradio_api/call/predict`
returns an `event_id`, then `GET /gradio_api/call/predict/<event_id>` streams the
result.

### Response

```jsonc
{
  "category": "bottle",
  "verdict": "defect",           // "pass" | "defect" | "uncertain"
  "message": null,               // set only for "uncertain"
  "anomaly_score": 61.85,
  "threshold": 31.44,
  "is_defective": true,
  "severity": 0.97,
  "defect_type": "broken_large",
  "defect_probabilities": { "broken_large": 0.999, "...": 0.001 },
  "defect_area_pct": 25.27,
  "images": { "original": "data:image/png;base64,…", "heatmap": "…", "outline": "…" },
  "latency_ms": 980.4
}
```

## Configuring the secret

The API refuses every call until a secret is set — a misconfigured Space fails
visibly rather than quietly serving an open model endpoint.

In the Space: **Settings → Variables and secrets → New secret**

| Name | Value |
| --- | --- |
| `DEFECT_DETECTOR_API_KEY` | any long random string |

The Space restarts automatically. The visible UI stays open either way; only the
API is gated.

> **What this protects against.** Casual scraping and idle traffic, not a
> determined attacker. Any client holding the key can use it freely, the key is
> visible to anyone who can read a caller's bundle, and Gradio's internal event
> endpoints remain reachable. It is right-sized for a portfolio demo and nothing
> more.

## Cold starts

Free Spaces sleep after a period of inactivity. The first request after a sleep
can take up to a minute while the container restarts and the first memory bank
loads. Subsequent requests take about a second.

## Limitations

- **No sense of "not my product".** The detector compares against one product's
  memory bank. A wildly different photo is caught and returned as `UNCERTAIN`
  (98% of cross-product photos in testing), but two similar products are not
  distinguished — pick the right one.
- **Defect type is a closed set.** A novel failure mode is still *detected*, but
  will be forced into one of that product's known labels.
- **Flagged area over-states.** The heatmap is upsampled and blurred, so the
  marked region reliably contains the defect without being a tight outline.

## License

Code: MIT. Models and sample images derive from the
[MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
(CC BY-NC-SA 4.0), so the shipped artefacts are **non-commercial** and carry
ShareAlike and attribution requirements.
