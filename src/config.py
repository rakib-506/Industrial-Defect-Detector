"""Central configuration for the industrial defect detection pipeline.

Everything is keyed by MVTec category. Nothing here hardcodes a defect-class
list or count: those are read from each category's `test/` subfolders at
runtime (see `dataset.defect_labels`).
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# anomalib's MVTec loader expects <root>/<category>/{train,test,ground_truth}.
DATASET_ROOT = Path(r"C:\Users\22301506\Documents\ComVi")
# Categories onboarded from client photo folders are materialised into the same
# layout here, so every downstream stage treats them identically to MVTec ones.
ONBOARDED_ROOT = PROJECT_ROOT / "data" / "onboarded"
# A handful of demo images per category, in the same layout, small enough to ship
# inside a container. Used only when the real dataset is not mounted - which is
# the case on Hugging Face Spaces, where the raw MVTec folders are excluded from
# the image. Populated by `python -m scripts.export_samples`.
SAMPLES_ROOT = PROJECT_ROOT / "deploy" / "samples"
# Onboarded first: an explicitly staged category must shadow a same-named folder
# in the MVTec root, or onboarding "wood" would silently train on MVTec's wood
# instead of the client's staged copy. Built-in names are refused at onboard
# time, so nothing can shadow the six shipped categories. Bundled samples come
# last, so a real dataset always wins when one is present.
DATASET_ROOTS = [ONBOARDED_ROOT, DATASET_ROOT, SAMPLES_ROOT]

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# The original MVTec categories. MVTec splits these into rigid objects and
# repeating textures; the distinction matters when reading results, so it is
# recorded rather than inferred. Onboarded categories declare their own type.
OBJECT_CATEGORIES = ["bottle", "capsule", "metal_nut"]
TEXTURE_CATEGORIES = ["carpet", "grid", "tile"]
BUILTIN_CATEGORIES = sorted(OBJECT_CATEGORIES + TEXTURE_CATEGORIES)
CATEGORIES = BUILTIN_CATEGORIES

DEFAULT_CATEGORY = "bottle"


def dataset_root_for(category: str) -> Path:
    """Which dataset root holds this category's images.

    Accepts a root that has either training images or a test split: the bundled
    demo samples ship only `test/`, since a deployed image serves the gallery but
    never trains.
    """
    for root in DATASET_ROOTS:
        base = root / category
        if (base / "train" / "good").is_dir() or (base / "test").is_dir():
            return root
    # Nothing on disk yet; fall back so error messages name the expected place.
    return DATASET_ROOT


def category_dir(category: str) -> Path:
    return dataset_root_for(category) / category


def trained_categories() -> list[str]:
    """Every category with a saved memory bank, builtin or onboarded."""
    if not MODELS_DIR.is_dir():
        return []
    return sorted(
        path.name for path in MODELS_DIR.iterdir() if (path / "patchcore.pt").exists()
    )


def is_onboarded(category: str) -> bool:
    return category not in BUILTIN_CATEGORIES


def category_type(category: str) -> str:
    """'object' or 'texture'. Onboarded categories record their own at onboard time."""
    if category in TEXTURE_CATEGORIES:
        return "texture"
    if category in OBJECT_CATEGORIES:
        return "object"
    marker = outputs_dir(category) / "category.json"
    if marker.exists():
        import json

        return json.loads(marker.read_text()).get("type", "object")
    return "object"


def models_dir(category: str) -> Path:
    return MODELS_DIR / category


def outputs_dir(category: str) -> Path:
    return OUTPUTS_DIR / category


def patchcore_ckpt(category: str) -> Path:
    return models_dir(category) / "patchcore.pt"


def classifier_path(category: str) -> Path:
    return models_dir(category) / "defect_classifier.joblib"


def metrics_path(category: str) -> Path:
    return outputs_dir(category) / "metrics.json"


def splits_path(category: str) -> Path:
    return outputs_dir(category) / "splits.json"


def feature_cache_path(category: str) -> Path:
    return outputs_dir(category) / "classifier_features.npz"


COMPARISON_PATH = OUTPUTS_DIR / "comparison.json"

# --- Image pipeline ----------------------------------------------------------
IMAGE_SIZE = (256, 256)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- PatchCore ---------------------------------------------------------------
BACKBONE = "wide_resnet50_2"
LAYERS = ["layer2", "layer3"]
CORESET_RATIO = 0.1
NUM_NEIGHBORS = 9
# Fraction of train/good held out of the memory bank, used to calibrate the
# decision threshold on images the detector has never memorised.
VAL_SPLIT_RATIO = 0.1

# Per-category PatchCore overrides. Empty means every category runs the same
# default settings; entries are added only where a measured comparison showed a
# real gain, and the reason is recorded alongside.
PATCHCORE_OVERRIDES: dict[str, dict] = {}


def patchcore_settings(category: str) -> dict:
    """Resolved PatchCore hyper-parameters for one category."""
    settings = {
        "backbone": BACKBONE,
        "layers": list(LAYERS),
        "coreset_ratio": CORESET_RATIO,
        "num_neighbors": NUM_NEIGHBORS,
    }
    settings.update(PATCHCORE_OVERRIDES.get(category, {}))
    return settings


# --- Out-of-distribution ("wrong product") check -----------------------------
# Fraction of the image that must be anomalous before a result is downgraded
# from DEFECT to UNCERTAIN.
#
# Why coverage and not a higher score. A genuine defect is *local*: the flaw
# covers part of the product and the rest still matches the memory bank. A photo
# of the wrong product is unfamiliar *everywhere*. Measured across all seven
# categories, the image score does NOT separate the two - genuine defects reach
# 85.3 on tile while wrong-product photos start at 55.1, so any score threshold
# either misses wrong products or misfires on real defects. Coverage separates
# cleanly on every category: the worst genuine defect covers 0.739 of the frame,
# the least-unfamiliar wrong product covers 0.753.
#
# 0.80 is a deliberate constant, not a fitted value: it encodes "a defect is
# local, so 80% of the frame being novel means this is not the product". The
# held-out good images are all at 0.000 coverage, so they offer no spread to fit
# against. Stored per category in metrics.json so it can be overridden.
OOD_COVERAGE_THRESHOLD = 0.80

# --- Defect-type classifier --------------------------------------------------
RANDOM_SEED = 42
CLASSIFIER_FOLDS = 5
DEFAULT_PCA_COMPONENTS = 40
CLASSIFIER_C = 0.2

# Cross-validation guards. StratifiedKFold needs at least `n_splits` members of
# every class, so a category whose rarest defect type falls below these bounds
# gets its estimate suppressed rather than reported from folds that cannot hold
# a full set of classes.
MIN_PER_CLASS_FOR_CV = CLASSIFIER_FOLDS
# Nested CV splits again inside each outer training fold, which retains roughly
# (1 - 1/folds) of each class, so the rarest class needs enough to survive twice.
MIN_PER_CLASS_FOR_NESTED_CV = int(round(CLASSIFIER_FOLDS / (1 - 1 / CLASSIFIER_FOLDS) + 0.5))
