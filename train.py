"""
Assignment 2: training pipeline.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
WEIGHTS_DIR = ROOT / "weights"
ARTIFACTS_DIR = ROOT / "artifacts"
RUNS_DIR = ROOT / "runs"


# =========================================================================== #
# PART 1 -- the cost / decision model.
#
# THE DECISION, IN ONE SENTENCE: a camera watches a work area on a
# construction site. For every snapshot, the system decides whether a worker
# is visible with no hard hat (a "head" box with no accompanying "helmet"
# box): PASS (nothing happens), ESCALATE (the snapshot goes to the site
# safety officer's phone for a quick look), or FAIL (the site alarm sounds
# immediately, no human in the loop).
#
# WHY A MISS AND A FALSE ALARM DO NOT COST THE SAME: missing a real
# violation (FN) means a worker keeps working unprotected until the next
# snapshot; the downside is a head injury. A false alarm (FP) costs a few
# seconds of a safety officer's attention, or in the FAIL band, one
# unnecessary alarm. Those are not close, which is the whole point of this
# assignment.
#
# EVERY NUMBER BELOW IS A SOURCED STARTING POINT, NOT A FINAL ANSWER.
# REPORT.md asks you to defend each one -- "I assumed X because Y, and the
# recommendation flips below Z" is the bar the brief sets. The ones marked
# JUDGMENT CALL below are the ones nobody can look up; sensitivity-test
# those first (README Step 6).
# =========================================================================== #

# --------------------------------------------------------------------------- #
# 1a. Dataset and training configuration
# --------------------------------------------------------------------------- #
WORKSPACE = "joseph-nelson"
PROJECT = "hard-hat-workers"
VERSION = 10  # raw_AllClasses: 70/20/10 split, head + helmet + person, unaugmented
ROBOFLOW_FORMAT = "yolov8"  # label layout Ultralytics reads; YOLO11 uses the same one

# A bare head with no overlapping helmet box: the violation this system
# exists to catch. Must be exactly one of the names in this dataset's data.yaml.
POSITIVE_CLASS = "head"

MODEL_CHECKPOINT = "yolo11s.pt"  # swap for "yolo11n.pt" for a smaller, faster model
EPOCHS = 60
IMG_SIZE = 640
BATCH = 16
SEED = 0

# How often the camera takes a snapshot of the work area. This is the
# exposure window a single missed violation is "live" for before the system
# gets another chance at it, and it is reused below in the FN-cost estimate
# and in the review-capacity math, so both sections agree on the same site.
SNAPSHOT_INTERVAL_SECONDS = 120

# --------------------------------------------------------------------------- #
# 1b. The cost matrix (EUR)
# --------------------------------------------------------------------------- #
# FN: the system says "compliant" and a worker is in fact bare-headed, for
# one snapshot interval, before the camera checks again.
#
#   cost = P(that exposure becomes a recordable injury) x (average cost of
#          that injury if it happens)
#
# The severity side is sourced: OSHA's $afety Pays calculator reports an
# average direct + indirect cost of USD 79,613 for a fracture claim (NCCI
# claims data; see osha.gov/safetypays/background), call it EUR 75,000 at a
# round conversion. Before you submit, pull the "head injury" / "concussion"
# category from the live calculator at https://www.osha.gov/safetypays
# instead of using the generic "fracture" figure here.
AVG_INJURY_COST_EUR = 75_000.00

# JUDGMENT CALL: nobody publishes "probability a single missed hard-hat
# detection becomes an injury in the next two minutes." 1-in-2,000 is an
# order-of-magnitude guess, not a measured figure, and it is the single
# number in this file most worth sweeping (README Step 6: "state the
# assumption that would most change your answer if it were wrong").
P_INJURY_IF_MISSED = 1 / 2000

COST_FN = AVG_INJURY_COST_EUR * P_INJURY_IF_MISSED  # EUR 37.50

# Employer cost of an hour of someone's time is their wage plus overhead
# (social security, benefits) -- a commonly used rule of thumb is to load the
# payslip rate by roughly 30% for this kind of internal costing.
LOADED_LABOUR_MULTIPLIER = 1.3

# FP: the system raises the alarm (FAIL band) over a compliant worker.
# Priced as a stopped task plus a supervisor's time to clear the false alarm.
SUPERVISOR_RATE_EUR_PER_HOUR = 25.00
FALSE_ALARM_MINUTES = 4
COST_FP = SUPERVISOR_RATE_EUR_PER_HOUR * LOADED_LABOUR_MULTIPLIER / 60 * FALSE_ALARM_MINUTES

# A human review (ESCALATE band): the safety officer glances at the snapshot.
REVIEW_SECONDS = 45
SAFETY_OFFICER_RATE_EUR_PER_HOUR = 22.00
COST_REVIEW = SAFETY_OFFICER_RATE_EUR_PER_HOUR * LOADED_LABOUR_MULTIPLIER / 3600 * REVIEW_SECONDS

# The officer is a second, differently-priced classifier, not an oracle.
# These are plausible for an easy visual check ("is there a hard hat here"),
# but vigilance for a rare target is well documented to decay with
# repetition (most escalated images will be false alarms), so treat these as
# optimistic and sensitivity-test them too.
SAFETY_OFFICER_RECALL = 0.97
SAFETY_OFFICER_SPECIFICITY = 0.95

# How often a snapshot genuinely shows a violation. JUDGMENT CALL, but not an
# unconstrained one: an 18-site Irish construction-safety study found
# external hard-hat non-compliance ranging from under 5% on well-run sites to
# over 60% on poorly-run ones. 0.10 sits on the well-managed end of that
# range. This is the single most site-specific number here -- replace it
# with real audit data from the site you are actually designing for, and it
# will move the chosen threshold more than any other assumption will.
BASE_RATE = 0.10

# --------------------------------------------------------------------------- #
# 1c. Review capacity (Step 7: how many people does the band you chose need?)
# --------------------------------------------------------------------------- #
SITE_CAMERAS = 5
SHIFT_HOURS = 10.0
SNAPSHOTS_PER_DAY = int(SITE_CAMERAS * SHIFT_HOURS * 3600 / SNAPSHOT_INTERVAL_SECONDS)


def review_capacity(
    review_share: float,
    snapshots_per_day: int = SNAPSHOTS_PER_DAY,
    review_seconds: int = REVIEW_SECONDS,
    shift_hours: float = SHIFT_HOURS,
) -> dict:
    """How many reviews/day a chosen band sends to a person, and how many
    safety officers that needs to clear them inside one shift."""
    reviews_per_day = snapshots_per_day * review_share
    officer_seconds_needed = reviews_per_day * review_seconds
    return {
        "snapshots_per_day": snapshots_per_day,
        "reviews_per_day": reviews_per_day,
        "officer_hours_per_day": officer_seconds_needed / 3600,
        "officers_needed": officer_seconds_needed / (shift_hours * 3600),
    }


# --------------------------------------------------------------------------- #
# 1d. Threshold grid swept by train.py and app.py.
# --------------------------------------------------------------------------- #
CONF_GRID: tuple[float, ...] = tuple(round(0.02 * i, 2) for i in range(1, 50))  # 0.02 .. 0.98


# --------------------------------------------------------------------------- #
# 1e. Per-image scoring. One "unit" is one snapshot, not one bounding box.
# --------------------------------------------------------------------------- #
@dataclass
class ScoredImage:
    image: str
    truth: str  # "violation" or "compliant"
    score: float  # max confidence among predicted POSITIVE_CLASS boxes; 0.0 if none


def zone_rates(rows: list[ScoredImage], low: float, high: float) -> tuple[float, float, float]:
    """Share of `rows` landing in PASS (< low), ESCALATE ([low, high)), FAIL (>= high)."""
    n = len(rows)
    if n == 0:
        return 0.0, 0.0, 0.0
    passed = sum(1 for r in rows if r.score < low) / n
    escalated = sum(1 for r in rows if low <= r.score < high) / n
    failed = sum(1 for r in rows if r.score >= high) / n
    return passed, escalated, failed


def three_way_cost(
    rows: list[ScoredImage],
    low: float,
    high: float,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
    cost_review: float = COST_REVIEW,
    officer_recall: float = SAFETY_OFFICER_RECALL,
    officer_specificity: float = SAFETY_OFFICER_SPECIFICITY,
    base_rate: float = BASE_RATE,
    n_units: int = 1000,
) -> dict:
    """Expected cost per `n_units` snapshots at a given [low, high) band.

    Zone rates are measured separately within the violation rows and the
    compliant rows, so they are properties of the model's score distribution
    and survive being re-projected onto a different base rate.
    """
    if low > high:
        low, high = high, low
    violations = [r for r in rows if r.truth == "violation"]
    compliant = [r for r in rows if r.truth == "compliant"]

    v_pass, v_review, _v_fail = zone_rates(violations, low, high)
    _c_pass, c_review, c_fail = zone_rates(compliant, low, high)

    n_violation = n_units * base_rate
    n_compliant = n_units * (1 - base_rate)

    # A missed violation: auto-passed, or escalated and waved through by the officer.
    fn = n_violation * (v_pass + v_review * (1 - officer_recall))
    # A false alarm: auto-failed, or escalated and the officer wrongly confirms it.
    fp = n_compliant * (c_fail + c_review * (1 - officer_specificity))
    reviews = n_violation * v_review + n_compliant * c_review

    return {
        "low": low,
        "high": high,
        "fn": fn,
        "fp": fp,
        "reviews": reviews,
        "review_share": reviews / n_units,
        "cost": fn * cost_fn + fp * cost_fp + reviews * cost_review,
    }


def rank_two_threshold_designs(
    rows: list[ScoredImage], grid: tuple[float, ...] = CONF_GRID, **kwargs
) -> list[dict]:
    """Every [low, high] band buildable from `grid`, cheapest first."""
    designs = [
        three_way_cost(rows, low, high, **kwargs)
        for i, low in enumerate(grid)
        for high in grid[i:]
    ]
    return sorted(designs, key=lambda d: d["cost"])


def best_single_threshold(
    rows: list[ScoredImage], grid: tuple[float, ...] = CONF_GRID, **kwargs
) -> dict:
    """Cheapest design with no review band (low == high): the model decides alone."""
    designs = [three_way_cost(rows, c, c, **kwargs) for c in grid]
    return min(designs, key=lambda d: d["cost"])


def human_checks_everything(rows: list[ScoredImage], **kwargs) -> dict:
    """Every snapshot goes to the safety officer."""
    return three_way_cost(rows, 0.0, 1.0 + 1e-9, **kwargs)


def do_nothing(
    base_rate: float = BASE_RATE, cost_fn: float = COST_FN, n_units: int = 1000
) -> dict:
    """No camera system at all: every violation in the population goes unmitigated."""
    fn = n_units * base_rate
    return {
        "low": None, "high": None, "fn": fn, "fp": 0.0,
        "reviews": 0.0, "review_share": 0.0, "cost": fn * cost_fn,
    }


def confusion_counts(rows: list[ScoredImage], low: float, high: float) -> dict:
    """Raw (not base-rate-adjusted) counts of how the test split's own images
    landed at this band -- the evidence the cost projection above is built on."""
    if low > high:
        low, high = high, low
    counts = {
        "violation": {"PASS": 0, "ESCALATE": 0, "FAIL": 0},
        "compliant": {"PASS": 0, "ESCALATE": 0, "FAIL": 0},
    }
    for r in rows:
        bucket = "PASS" if r.score < low else "ESCALATE" if r.score < high else "FAIL"
        counts[r.truth][bucket] += 1
    return counts


# =========================================================================== #
# PART 2 -- the training pipeline.
# =========================================================================== #

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def _ensure_absolute_path_key(data_yaml: Path, dataset_dir: Path) -> None:
    """Roboflow's exported data.yaml gives train/val/test as paths relative to
    *itself* (e.g. `train: ../train/images`), but Ultralytics resolves those
    relative paths against the current working directory at train time, not
    against the yaml's own location. Since main.py runs this script with
    cwd=ROOT, that mismatch is real here, not theoretical. Writing an
    absolute `path:` key removes the ambiguity regardless of cwd."""
    spec = yaml.safe_load(data_yaml.read_text())
    if spec.get("path") != str(dataset_dir):
        spec["path"] = str(dataset_dir)
        data_yaml.write_text(yaml.safe_dump(spec, sort_keys=False))


def download_dataset() -> Path:
    """Download the dataset version named above, unless it's already here."""
    data_yaml = DATASET_DIR / "data.yaml"
    if data_yaml.exists():
        print(f"Found an existing dataset at {DATASET_DIR}, skipping download.")
        _ensure_absolute_path_key(data_yaml, DATASET_DIR)
        return DATASET_DIR

    api_key = load_api_key()
    try:
        rf = Roboflow(api_key=api_key)
        version = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
        version.download(ROBOFLOW_FORMAT, location=str(DATASET_DIR), overwrite=True)
    except Exception as error:  # noqa: BLE001 -- a clear reason beats a raw traceback
        raise SystemExit(
            f"Could not download {WORKSPACE}/{PROJECT} v{VERSION} from Roboflow: "
            f"{type(error).__name__}: {error}\n"
            f"Check your key with `uv run python api_key.py`, and that "
            f"{WORKSPACE}/{PROJECT} v{VERSION} is still public on Roboflow Universe."
        ) from error

    _ensure_absolute_path_key(data_yaml, DATASET_DIR)
    return DATASET_DIR


def load_class_names(data_yaml: Path) -> list[str]:
    spec = yaml.safe_load(data_yaml.read_text())
    names = spec["names"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def finetune(data_yaml: Path) -> tuple[YOLO, Path]:
    print(f"Fine-tuning {MODEL_CHECKPOINT} on {data_yaml} (device={DEVICE}) ...")
    model = YOLO(MODEL_CHECKPOINT)
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        project=str(RUNS_DIR),
        name="finetune",
        exist_ok=True,
        seed=SEED,
    )
    run_dir = Path(model.trainer.save_dir)
    return YOLO(run_dir / "weights" / "best.pt"), run_dir


def ground_truth_label(label_path: Path, class_names: list[str]) -> set[str]:
    """YOLO label file -> the set of class names with at least one box in this image."""
    if not label_path.exists():
        return set()
    ids = {
        int(line.split()[0])
        for line in label_path.read_text().strip().splitlines()
        if line.strip()
    }
    return {class_names[i] for i in ids}


def score_split(model: YOLO, class_names: list[str], split: str = "test") -> list[ScoredImage]:
    """One row per image: ground truth ("violation"/"compliant") and the model's
    best score for the positive class, read at a near-zero confidence floor so
    every threshold in CONF_GRID can be swept without re-running inference.
    """
    images_dir = DATASET_DIR / split / "images"
    labels_dir = DATASET_DIR / split / "labels"
    image_paths = sorted(
        p for p in images_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    rows: list[ScoredImage] = []
    for image_path in image_paths:
        labels = ground_truth_label(labels_dir / f"{image_path.stem}.txt", class_names)
        truth = "violation" if POSITIVE_CLASS in labels else "compliant"

        result = model.predict(str(image_path), conf=0.001, device=DEVICE, verbose=False)[0]
        positive_scores = [
            float(box.conf)
            for box in result.boxes
            if class_names[int(box.cls)] == POSITIVE_CLASS
        ]
        score = max(positive_scores) if positive_scores else 0.0
        rows.append(ScoredImage(image=str(image_path.relative_to(ROOT)), truth=truth, score=score))
    return rows


def zero_shot_examples(split: str = "test", n: int = 8) -> list[dict]:
    """What the unmodified, pretrained COCO model calls a handful of test
    images. COCO has no `head` or `helmet` class, so it cannot produce the
    positive class at any threshold -- this just shows what it reaches for
    instead (see README Step 4 and the vision_api_inspection notebook)."""
    base = YOLO("yolo11n.pt")
    images_dir = DATASET_DIR / split / "images"
    sample = sorted(images_dir.glob("*"))[:n]
    examples = []
    for image_path in sample:
        result = base.predict(str(image_path), conf=0.25, device=DEVICE, verbose=False)[0]
        calls = sorted({base.names[int(b.cls)] for b in result.boxes})
        examples.append({"image": image_path.name, "coco_calls": calls})
    return examples


def per_class_metrics(metrics, class_names: list[str]) -> list[dict]:
    """Per-class precision, recall, AP50 and AP50-95. Ultralytics' exact
    attribute names have moved between versions; if this raises an
    AttributeError, run `print(dir(metrics.box))` and adjust the names below.
    """
    box = metrics.box
    rows = []
    for idx, class_id in enumerate(box.ap_class_index):
        rows.append(
            {
                "class": class_names[int(class_id)],
                "precision": float(box.p[idx]),
                "recall": float(box.r[idx]),
                "AP50": float(box.ap50[idx]),
                "AP50_95": float(box.ap[idx]),
            }
        )
    return rows


def main() -> None:
    load_api_key()  # fail fast, with a clear message, if the key is missing
    dataset_dir = download_dataset()
    data_yaml = dataset_dir / "data.yaml"
    class_names = load_class_names(data_yaml)
    if POSITIVE_CLASS not in class_names:
        raise SystemExit(
            f"'{POSITIVE_CLASS}' is not one of this dataset's classes: "
            f"{class_names}. Update POSITIVE_CLASS above."
        )

    finetuned, run_dir = finetune(data_yaml)

    WEIGHTS_DIR.mkdir(exist_ok=True)
    best_weights = WEIGHTS_DIR / "best.pt"
    shutil.copy(run_dir / "weights" / "best.pt", best_weights)
    print(f"Committed weights to {best_weights}")

    print("Evaluating on the test split ...")
    test_metrics = finetuned.val(
        data=str(data_yaml), split="test", device=DEVICE,
        project=str(RUNS_DIR), name="test_eval", exist_ok=True,
    )
    overall_metrics = {
        "precision": test_metrics.results_dict.get("metrics/precision(B)"),
        "recall": test_metrics.results_dict.get("metrics/recall(B)"),
        "mAP50": test_metrics.results_dict.get("metrics/mAP50(B)"),
        "mAP50_95": test_metrics.results_dict.get("metrics/mAP50-95(B)"),
    }
    per_class = per_class_metrics(test_metrics, class_names)

    print("Scoring the test split for the cost model ...")
    rows = score_split(finetuned, class_names, split="test")

    print("Checking what the zero-shot COCO model calls instead ...")
    zero_shot = zero_shot_examples()

    print("Sweeping the confidence threshold ...")
    single_threshold_sweep = [three_way_cost(rows, c, c) for c in CONF_GRID]
    ranked = rank_two_threshold_designs(rows)
    best = ranked[0]
    single_best = best_single_threshold(rows)
    everything = human_checks_everything(rows)
    nothing = do_nothing()
    capacity = review_capacity(best["review_share"])

    print(f"\nChosen operating point: ESCALATE >= {best['low']}, FAIL >= {best['high']}")
    print(f"  EUR {best['cost']:,.2f} / 1,000 snapshots, {best['review_share']:.1%} reviewed")
    print(f"  model-alone best single cut: EUR {single_best['cost']:,.2f} (cut {single_best['high']})")
    print(f"  officer checks everything:   EUR {everything['cost']:,.2f}")
    print(f"  no system at all:            EUR {nothing['cost']:,.2f}")
    print(
        f"  review load: {capacity['reviews_per_day']:.0f} reviews/day, "
        f"needs {capacity['officers_needed']:.2f} safety officer(s)"
    )

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    artifact = {
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION},
        "classes": class_names,
        "positive_class": POSITIVE_CLASS,
        "device_used_for_training": DEVICE,
        "model_checkpoint": MODEL_CHECKPOINT,
        "overall_metrics": overall_metrics,
        "per_class_metrics": per_class,
        "baselines": {
            "zero_shot_coco_examples": zero_shot,
            "zero_shot_note": (
                "COCO has no 'head' or 'helmet' class, so the pretrained model "
                "scores 0 on the positive class at every threshold: in this cost "
                "model that is mathematically identical to the predict-nothing "
                "baseline below."
            ),
            "predict_nothing": nothing,
        },
        "cost_matrix": {
            "cost_fn": COST_FN,
            "cost_fp": COST_FP,
            "cost_review": COST_REVIEW,
            "base_rate": BASE_RATE,
            "safety_officer_recall": SAFETY_OFFICER_RECALL,
            "safety_officer_specificity": SAFETY_OFFICER_SPECIFICITY,
            "avg_injury_cost_eur": AVG_INJURY_COST_EUR,
            "p_injury_if_missed": P_INJURY_IF_MISSED,
        },
        "cost_sweep_single_threshold": single_threshold_sweep,
        "chosen_operating_point": best,
        "comparison": {
            "three_way": best,
            "model_alone": single_best,
            "officer_checks_everything": everything,
            "do_nothing": nothing,
        },
        "review_capacity": capacity,
        "test_rows": [row.__dict__ for row in rows],
    }
    (ARTIFACTS_DIR / "results.json").write_text(json.dumps(artifact, indent=2))
    print(f"\nSaved {ARTIFACTS_DIR / 'results.json'} for app.py.")


if __name__ == "__main__":
    main()