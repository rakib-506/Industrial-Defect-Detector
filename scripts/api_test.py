"""Exercise every API route against a running server.

Run:  python -m scripts.api_test   (server must be up on :8000)
"""

from __future__ import annotations

import sys

import httpx

from src import dataset

import os

# Endpoints live under /api so the same app can serve the built UI at /.
BASE = os.environ.get("DEFECT_DETECTOR_API", "http://127.0.0.1:8000") + "/api"


def main() -> int:
    failures: list[str] = []

    with httpx.Client(base_url=BASE, timeout=180.0) as client:
        health = client.get("/health").json()
        print(f"GET  /health     -> status={health.get('status')} "
              f"categories={len(health.get('categories', []))}")
        if health.get("status") != "ok":
            failures.append("health not ok")

        cats = client.get("/categories").json()
        print(f"GET  /categories -> {len(cats)} categories")

        def fmt(value, spec=".4f"):
            # Onboarded categories legitimately report null metrics when they
            # arrive without labelled defects or ground-truth masks.
            return "n/a" if value is None else format(value, spec)

        for c in cats:
            clf = c["metrics"].get("classifier") or {}
            acc = clf.get("nested_cv_accuracy_mean") or clf.get("cv_accuracy_mean")
            tag = " (onboarded)" if c.get("onboarded") else ""
            print(
                f"     {c['category']:<16} {c['type']:<8} "
                f"img_auroc={fmt(c['metrics']['image_auroc'])} "
                f"px_auroc={fmt(c['metrics']['pixel_auroc'])} "
                f"classes={len(c['defect_classes'])} "
                f"clf={fmt(acc, '.3f')}{tag}"
            )
            # A category with a classifier must expose the classes it predicts;
            # one without a classifier must expose none.
            if c["has_classifier"] and not c["defect_classes"]:
                failures.append(f"{c['category']}: has a classifier but no defect classes")
            if not c["has_classifier"] and c["defect_classes"]:
                failures.append(f"{c['category']}: no classifier but reports defect classes")

        comparison = client.get("/comparison").json()
        print(f"GET  /comparison -> {len(comparison)} rows")
        if len(comparison) != len(cats):
            failures.append(
                f"comparison rows ({len(comparison)}) != categories ({len(cats)})"
            )

        # Per category: metadata, samples, and one good + one defective sample.
        for c in cats:
            category = c["category"]
            meta = client.get(f"/metadata?category={category}").json()
            if meta["category"] != category:
                failures.append(f"{category}: metadata returned {meta['category']}")

            samples = client.get(f"/samples?category={category}").json()
            if not samples:
                failures.append(f"{category}: no samples")
                continue
            if any(not s["thumbnail"].startswith("data:image/") for s in samples):
                failures.append(f"{category}: bad thumbnail")

            checked = 0
            seen: set[str] = set()
            for s in samples:
                if s["label"] in seen:
                    continue
                seen.add(s["label"])
                if s["label"] != "good" and checked >= 2:
                    continue
                r = client.post(f"/predict/sample/{s['id']}").json()
                expected = s["label"] != "good"
                checked += 1
                if r["category"] != category:
                    failures.append(f"{s['id']}: wrong category in response")

                # A category with no classifier must never assert a defect type,
                # even on an image it flags as defective.
                if not meta["has_classifier"]:
                    if r["defect_type"] is not None or r["defect_probabilities"]:
                        failures.append(
                            f"{s['id']}: no classifier, but got defect_type "
                            f"{r['defect_type']!r}"
                        )
                elif r["is_defective"] and not r["defect_probabilities"]:
                    failures.append(f"{s['id']}: flagged defective but no probabilities")

                # Verdict accuracy is a model property, not an API contract.
                # Onboarded categories may be deliberately under-calibrated and
                # already carry an explicit warning from the onboarding script.
                if r["is_defective"] != expected and not meta.get("onboarded"):
                    failures.append(f"{s['id']}: detection mismatch")
            print(f"POST /predict/sample  {category:<11} -> {checked} samples checked")

        # A real multipart upload, routed to a non-default category.
        target = cats[-1]["category"]
        defect = next(
            s for s in dataset.list_split(target, "test") if s.is_defective
        )
        with open(defect.path, "rb") as fh:
            r = client.post(
                "/predict",
                files={"file": (defect.path.name, fh, "image/png")},
                data={"category": target},
            )
        body = r.json()
        print(
            f"POST /predict (upload {target}/{defect.label}/{defect.path.name}) -> "
            f"status={r.status_code} score={body['anomaly_score']:.2f} "
            f"type={body['defect_type']} latency={body['latency_ms']:.0f}ms"
        )
        if r.status_code != 200 or not body["is_defective"]:
            failures.append("upload prediction failed")
        for key in ("original", "heatmap", "outline"):
            if not body["images"][key].startswith("data:image/"):
                failures.append(f"bad {key} image")

        # The wrong-product safety check, over HTTP: a clean photo of one
        # category sent to a different category's checker must come back
        # "uncertain" with no defect type, not a confident DEFECT.
        if len(cats) >= 2:
            checker = cats[0]["category"]
            other = next(c["category"] for c in cats if c["category"] != checker)
            wrong = next(
                s for s in dataset.list_split(other, "test") if not s.is_defective
            )
            with open(wrong.path, "rb") as fh:
                r = client.post(
                    "/predict",
                    files={"file": (wrong.path.name, fh, "image/png")},
                    data={"category": checker},
                )
            body = r.json()
            print(
                f"POST /predict (clean {other} -> {checker} checker) -> "
                f"verdict={body['verdict']} type={body['defect_type']} "
                f"coverage={body['defect_area_pct']:.0f}%"
            )
            if body["verdict"] != "uncertain":
                failures.append(
                    f"wrong-product check: expected 'uncertain', got {body['verdict']!r}"
                )
            if body["defect_type"] is not None or body["defect_probabilities"]:
                failures.append("wrong-product check: uncertain result carried a defect type")
            if not body["message"]:
                failures.append("wrong-product check: no message returned")
            if not body["images"]["heatmap"].startswith("data:image/"):
                failures.append("wrong-product check: heatmap missing")

        # Error paths.
        bad = client.post("/predict", files={"file": ("x.png", b"nope", "image/png")})
        print(f"POST /predict (garbage)            -> status={bad.status_code}")
        if bad.status_code != 400:
            failures.append(f"expected 400 for garbage upload, got {bad.status_code}")

        missing = client.post("/predict/sample/bottle__nope__000")
        print(f"POST /predict/sample (unknown)     -> status={missing.status_code}")
        if missing.status_code != 404:
            failures.append(f"expected 404 for unknown sample, got {missing.status_code}")

        unknown_cat = client.get("/metadata?category=banana")
        print(f"GET  /metadata (unknown category)  -> status={unknown_cat.status_code}")
        if unknown_cat.status_code != 404:
            failures.append(
                f"expected 404 for unknown category, got {unknown_cat.status_code}"
            )

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nall API checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
