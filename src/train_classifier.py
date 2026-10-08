"""Train the defect-type explainability classifier for one category.

PatchCore says *whether* a part is defective; it cannot say what went wrong,
because it never sees a defect during training. This second stage uses the
labelled MVTec test subfolders to name the defect once one has been localised.
The class list is read from those folders, so categories with 3, 4 or 5 defect
types all work without changing anything here.

Labelled defects are scarce - 11 images for the rarest `grid` class - so the
design leans on regularisation rather than capacity, and refuses to report an
accuracy it cannot support. See `config.MIN_PER_CLASS_FOR_CV`.

Run:  python -m src.train_classifier --category bottle --nested-cv
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config, dataset, features
from .estimators import SafePCA
from .imaging import load_image, preprocess
from .patchcore_infer import PatchCoreInferencer

APPEARANCE_DIM = 1536  # wide_resnet50_2 layer2 (512) + layer3 (1024)


def build_pipeline(appearance_dim: int, n_components: int, C: float = config.CLASSIFIER_C) -> Pipeline:
    """Two feature blocks, scaled separately; only appearance gets compressed."""
    pre = ColumnTransformer(
        [
            (
                "appearance",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("pca", SafePCA(n_components=n_components, random_state=config.RANDOM_SEED)),
                    ]
                ),
                slice(0, appearance_dim),
            ),
            ("geometry", StandardScaler(), slice(appearance_dim, None)),
        ]
    )
    return Pipeline(
        [
            ("features", pre),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=config.RANDOM_SEED,
                ),
            ),
        ]
    )


def extract_dataset(
    category: str,
    inferencer: PatchCoreInferencer,
    pixel_threshold: float,
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Featurise every labelled defective image in the category's test split.

    Cached to disk keyed on the pixel threshold, since a forward pass per image
    dominates the runtime of any hyper-parameter search.
    """
    cache = config.feature_cache_path(category)
    if use_cache and cache.exists():
        cached = np.load(cache, allow_pickle=True)
        if float(cached["pixel_threshold"]) == pixel_threshold:
            X, y = cached["X"], cached["y"].astype(str)
            print(f"[clf:{category}] loaded cached features {X.shape}")
            return X, y, sorted(set(y.tolist()))

    samples = [s for s in dataset.list_split(category, "test") if s.is_defective]
    if not samples:
        raise SystemExit(f"No labelled defect images found under {category}/test/.")

    vectors: list[np.ndarray] = []
    labels: list[str] = []
    for index, sample in enumerate(samples, start=1):
        batch = preprocess(load_image(sample.path))
        result = inferencer.predict(batch)
        embedding = inferencer.embed(batch)
        appearance, geometry = features.build_feature_vector(
            embedding, result.anomaly_map, pixel_threshold
        )
        vectors.append(np.concatenate([appearance, geometry]))
        labels.append(sample.label)
        if index % 25 == 0 or index == len(samples):
            print(f"[clf:{category}] featurised {index}/{len(samples)}")

    X = np.vstack(vectors).astype(np.float64)
    y = np.array(labels)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y, pixel_threshold=pixel_threshold)
    return X, y, sorted(set(labels))


def run_nested_cv(X: np.ndarray, y: np.ndarray, folds: int) -> tuple[float, float]:
    """Unbiased estimate: the hyper-parameter search runs inside each fold."""
    grid = {
        "features__appearance__pca__n_components": [12, 20, 30, 40],
        "clf__C": [0.05, 0.2, 1.0, 5.0],
    }
    outer = RepeatedStratifiedKFold(n_splits=folds, n_repeats=3, random_state=config.RANDOM_SEED)
    scores: list[float] = []
    for train_idx, test_idx in outer.split(X, y):
        search = GridSearchCV(
            build_pipeline(APPEARANCE_DIM, config.DEFAULT_PCA_COMPONENTS),
            grid,
            cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_SEED),
            n_jobs=-1,
        )
        search.fit(X[train_idx], y[train_idx])
        scores.append(accuracy_score(y[test_idx], search.predict(X[test_idx])))
    return float(np.mean(scores)), float(np.std(scores))


def train_category(
    category: str,
    pca_components: int = config.DEFAULT_PCA_COMPONENTS,
    folds: int = config.CLASSIFIER_FOLDS,
    nested: bool = False,
    quiet: bool = False,
) -> dict:
    def log(message: str) -> None:
        if not quiet:
            print(message)

    metrics_path = config.metrics_path(category)
    if not metrics_path.exists():
        raise SystemExit(
            f"Missing {metrics_path} - run `python -m src.calibrate --category {category}` first."
        )
    metrics = json.loads(metrics_path.read_text())
    pixel_threshold = float(metrics["pixel_threshold"])

    # An onboarded category may arrive with no labelled defects. That is a
    # supported state, not an error: stage 1 still detects and localises, and
    # the API simply returns no defect type. Record why and stop here.
    if not dataset.defect_labels(category):
        result = {
            "classes": [],
            "class_counts": {},
            "n_samples": 0,
            "trained": False,
            "cv_status": "no labelled defect examples - classifier not trained",
            "nested_cv_status": "no labelled defect examples - classifier not trained",
            "cv_accuracy_mean": None,
            "nested_cv_accuracy_mean": None,
        }
        log(f"[clf:{category}] no labelled defects found - skipping classifier")
        metrics["classifier"] = result
        metrics_path.write_text(json.dumps(metrics, indent=2))
        # Remove any stale classifier so serving cannot pick up a previous one.
        config.classifier_path(category).unlink(missing_ok=True)
        return result

    inferencer = PatchCoreInferencer(category)
    log(f"[clf:{category}] device={inferencer.device}, pixel_threshold={pixel_threshold:.4f}")

    X, y, classes = extract_dataset(category, inferencer, pixel_threshold)
    counts = {name: int((y == name).sum()) for name in classes}
    min_per_class = min(counts.values())
    log(f"[clf:{category}] features {X.shape}, {len(classes)} defect classes")
    for name, n in counts.items():
        log(f"       {name}: {n} images")

    # --- Decide what can honestly be reported --------------------------------
    # StratifiedKFold needs at least `n_splits` members of the rarest class.
    usable_folds = min(folds, min_per_class)
    cv_possible = usable_folds >= 2
    cv_reliable = min_per_class >= config.MIN_PER_CLASS_FOR_CV
    nested_possible = min_per_class >= config.MIN_PER_CLASS_FOR_NESTED_CV

    result: dict = {
        "classes": classes,
        "class_counts": counts,
        "n_samples": int(len(y)),
        "min_per_class": min_per_class,
        "pca_components": pca_components,
        "C": config.CLASSIFIER_C,
        "folds": usable_folds if cv_possible else None,
        "cv_accuracy_mean": None,
        "cv_accuracy_std": None,
        "nested_cv_accuracy_mean": None,
        "nested_cv_accuracy_std": None,
        "confusion_matrix": None,
        "cv_status": "ok",
        "nested_cv_status": "ok",
        "trained": True,
    }

    if not cv_possible:
        result["cv_status"] = (
            f"insufficient samples for CV: rarest class has {min_per_class} image(s), "
            "need at least 2"
        )
        result["nested_cv_status"] = result["cv_status"]
        log(f"[clf:{category}] {result['cv_status']}")
    else:
        if not cv_reliable:
            result["cv_status"] = (
                f"reduced to {usable_folds}-fold: rarest class has {min_per_class} "
                f"image(s), below the {config.MIN_PER_CLASS_FOR_CV} needed for "
                f"{folds}-fold - treat as indicative only"
            )
            log(f"[clf:{category}] WARNING {result['cv_status']}")

        cv = RepeatedStratifiedKFold(
            n_splits=usable_folds, n_repeats=6, random_state=config.RANDOM_SEED
        )
        fold_accuracies = []
        for train_idx, test_idx in cv.split(X, y):
            model = build_pipeline(APPEARANCE_DIM, pca_components)
            model.fit(X[train_idx], y[train_idx])
            fold_accuracies.append(accuracy_score(y[test_idx], model.predict(X[test_idx])))
        result["cv_accuracy_mean"] = float(np.mean(fold_accuracies))
        result["cv_accuracy_std"] = float(np.std(fold_accuracies))

        single_cv = StratifiedKFold(
            n_splits=usable_folds, shuffle=True, random_state=config.RANDOM_SEED
        )
        y_oof = cross_val_predict(
            build_pipeline(APPEARANCE_DIM, pca_components), X, y, cv=single_cv
        )
        result["confusion_matrix"] = confusion_matrix(y, y_oof, labels=classes).tolist()

        log(
            f"\n[clf:{category}] cross-validated accuracy: "
            f"{result['cv_accuracy_mean']:.4f} +/- {result['cv_accuracy_std']:.4f} "
            f"({usable_folds}-fold x 6 repeats)"
        )
        if not quiet:
            print(classification_report(y, y_oof, labels=classes, zero_division=0))
            header = " " * 16 + "".join(f"{c[:14]:>17}" for c in classes)
            print("[clf] confusion matrix (rows = truth, cols = predicted)")
            print(header)
            for name, row in zip(classes, result["confusion_matrix"]):
                print(f"{name[:15]:>15} " + "".join(f"{v:>17}" for v in row))

    if nested:
        if not nested_possible:
            result["nested_cv_status"] = (
                f"insufficient samples for nested CV: rarest class has {min_per_class} "
                f"image(s), need at least {config.MIN_PER_CLASS_FOR_NESTED_CV}"
            )
            log(f"[clf:{category}] {result['nested_cv_status']}")
        else:
            mean, std = run_nested_cv(X, y, usable_folds)
            result["nested_cv_accuracy_mean"] = mean
            result["nested_cv_accuracy_std"] = std
            log(f"\n[clf:{category}] nested CV accuracy: {mean:.4f} +/- {std:.4f}")
    else:
        result["nested_cv_status"] = "not run"

    # Refit on everything for the shipped artefact.
    pipeline = build_pipeline(APPEARANCE_DIM, pca_components)
    pipeline.fit(X, y)
    config.models_dir(category).mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "classes": list(pipeline.classes_),
            "pixel_threshold": pixel_threshold,
            "appearance_dim": APPEARANCE_DIM,
            "category": category,
        },
        config.classifier_path(category),
    )
    log(f"\n[clf:{category}] saved -> {config.classifier_path(category)}")

    metrics["classifier"] = result
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default=config.DEFAULT_CATEGORY)
    parser.add_argument("--pca-components", type=int, default=config.DEFAULT_PCA_COMPONENTS)
    parser.add_argument("--folds", type=int, default=config.CLASSIFIER_FOLDS)
    parser.add_argument(
        "--nested-cv",
        action="store_true",
        help="Also run nested CV, which re-selects hyper-parameters inside each "
        "outer fold and so is not biased by the tuning already done.",
    )
    args = parser.parse_args()
    train_category(args.category, args.pca_components, args.folds, args.nested_cv)


if __name__ == "__main__":
    main()
