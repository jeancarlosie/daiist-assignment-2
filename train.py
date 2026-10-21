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

MODEL_CHECKPOINT = "yolo11n.pt"  # nano: fastest to fine-tune; swap for "yolo11s.pt" if you have a GPU
EPOCHS = 25  # paired with PATIENCE below; 3 large, visually distinct classes don't need 60
IMG_SIZE = 416  # smaller than the usual 640 -- compute scales roughly with imgsz^2
BATCH = 16
PATIENCE = 15  # stop early if val mAP hasn't improved in this many epochs
SEED = 0

# Confidence floor for validation's internal NMS pass (used both during each
# epoch's automatic validation and the final test-split eval). Ultralytics
# defaults this near 0, which lets an early, noisy model flood NMS with
# thousands of near-zero-confidence candidate boxes per image -- on MPS this
# can make a single epoch's validation take 10x+ longer than training it did
# (watch for repeated "NMS time limit exceeded" warnings as the symptom).
# 0.01 is a 10x cut in candidate volume with a negligible effect on mAP, since
# real detections score well above it even early in training.
VAL_CONF = 0.01

# How often the camera takes a snapshot of the work area. This feeds the
# review-capacity math below (snapshots/day). It does NOT feed COST_FN
# numerically -- P_INJURY_IF_MISSED there is a flat judgment call, not
# derived from this interval. If you want the two to agree explicitly in
# REPORT.md, say how this interval bounds the exposure window you assumed
# when picking P_INJURY_IF_MISSED, rather than treating them as already linked.
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


def image_level_metrics(rows: list[ScoredImage], cut: float) -> dict:
    """Snapshot-level precision / recall / accuracy if every score >= cut is
    flagged as a violation. This is the unit the decision is made on, so it is
    the fair way to put the finetuned model and both baselines side by side."""
    tp = sum(1 for r in rows if r.truth == "violation" and r.score >= cut)
    fp = sum(1 for r in rows if r.truth == "compliant" and r.score >= cut)
    fn = sum(1 for r in rows if r.truth == "violation" and r.score < cut)
    tn = sum(1 for r in rows if r.truth == "compliant" and r.score < cut)
    n = len(rows) or 1
    return {
        "cut": cut, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "accuracy": (tp + tn) / n,
    }


def sensitivity_table(rows: list[ScoredImage]) -> list[dict]:
    """Re-choose the band while moving one assumption at a time (README Step 6:
    'the answer changes if X drops below Z'). The miss cost is moved through
    P_INJURY_IF_MISSED, the judgment call it rests on; the base rate is the
    most site-specific number in the file."""
    scenarios = [
        ("p_injury_if_missed", P_INJURY_IF_MISSED * m, {"cost_fn": COST_FN * m})
        for m in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
    ] + [
        ("base_rate", b, {"base_rate": b})
        for b in (0.02, 0.05, 0.10, 0.20, 0.40)
    ]
    table = []
    for assumption, value, overrides in scenarios:
        best = rank_two_threshold_designs(rows, **overrides)[0]
        alone = best_single_threshold(rows, **overrides)
        everything = human_checks_everything(rows, **overrides)
        nothing = do_nothing(
            base_rate=overrides.get("base_rate", BASE_RATE),
            cost_fn=overrides.get("cost_fn", COST_FN),
        )
        table.append({
            "assumption": assumption, "value": value,
            "low": best["low"], "high": best["high"],
            "three_way_cost": best["cost"], "review_share": best["review_share"],
            "model_alone_cost": alone["cost"],
            "officer_checks_everything_cost": everything["cost"],
            "do_nothing_cost": nothing["cost"],
        })
    return table


# =========================================================================== #
# PART 2 -- the training pipeline.
# =========================================================================== #

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Standalone (post-training) validation/prediction calls have repeatedly
# stalled for hours on MPS due to NMS, even with VAL_CONF narrowing the
# candidate pool -- unlike the per-epoch validation inside model.train(),
# which stayed fast. These are one-off calls, not repeated across 25 epochs,
# so trading MPS's forward-pass speed for a CPU path that doesn't hit this
# problem is worth it specifically here. Training itself stays on DEVICE.
EVAL_DEVICE = "cpu" if DEVICE == "mps" else DEVICE


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
    # Ultralytics pins dataloader workers to 0 on MPS (a known macOS
    # multiprocessing issue with the MPS backend), so all image loading and
    # augmentation runs single-threaded on the CPU. Mosaic augmentation
    # (compositing 4 images into one per sample) is the heaviest part of that
    # pipeline. On MPS/CPU, drop it and shrink the batch so one epoch actually
    # finishes in a practical time; a real GPU keeps the full settings.
    if DEVICE in ("mps", "cpu"):
        batch, mosaic = 8, 0.0
        print(f"Training on {DEVICE}: using batch={batch}, mosaic={mosaic} to keep this practical.")
    else:
        batch, mosaic = BATCH, 1.0

    print(f"Fine-tuning {MODEL_CHECKPOINT} on {data_yaml} (device={DEVICE}) ...")
    model = YOLO(MODEL_CHECKPOINT)
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=batch,
        mosaic=mosaic,
        patience=PATIENCE,
        conf=VAL_CONF,
        cache="disk",  # cache each image's decode/resize once instead of redoing
                        # it from the JPEG on every single epoch
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


def score_split(
    model: YOLO,
    class_names: list[str],
    target_class: str = POSITIVE_CLASS,
    split: str = "test",
    chunk_size: int = 32,
) -> list[ScoredImage]:
    """One row per test image: the ground truth from the dataset's labels
    ("violation" if any `head` box, else "compliant"), and the model's highest
    confidence for `target_class` -- looked up in the model's OWN vocabulary,
    so the same function scores the finetuned model ("head") and the COCO
    zero-shot baseline ("person", the closest thing COCO has).

    Scored at VAL_CONF rather than near zero: CONF_GRID starts at 0.02, so a
    score below 0.01 lands in PASS at every threshold anyway, and a near-zero
    floor floods NMS. Predicts in chunks: same results, less per-call overhead.
    """
    images_dir = DATASET_DIR / split / "images"
    labels_dir = DATASET_DIR / split / "labels"
    image_paths = sorted(
        p for p in images_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    rows: list[ScoredImage] = []
    total = len(image_paths)
    for start in range(0, total, chunk_size):
        chunk = image_paths[start : start + chunk_size]
        print(f"  scoring {min(start + chunk_size, total)}/{total} ...")
        results = model.predict(
            [str(p) for p in chunk], conf=VAL_CONF, imgsz=IMG_SIZE,
            device=EVAL_DEVICE, batch=len(chunk), verbose=False,
        )
        for image_path, result in zip(chunk, results):
            labels = ground_truth_label(labels_dir / f"{image_path.stem}.txt", class_names)
            truth = "violation" if POSITIVE_CLASS in labels else "compliant"
            target_scores = [
                float(box.conf)
                for box in result.boxes
                if model.names[int(box.cls)] == target_class
            ]
            score = max(target_scores) if target_scores else 0.0
            rows.append(
                ScoredImage(image=str(image_path.relative_to(ROOT)), truth=truth, score=score)
            )
    return rows


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


def detection_confusion_matrix(metrics, class_names: list[str]) -> dict | None:
    """Box-level confusion matrix from the test-split evaluation: which class
    the model confuses with which (e.g. a bare head read as a helmet), plus
    'background' for missed objects and false detections. Ultralytics stores
    it as matrix[predicted, true]. Returns None if this Ultralytics version
    does not expose it on the returned metrics object."""
    matrix = getattr(getattr(metrics, "confusion_matrix", None), "matrix", None)
    if matrix is None:
        print("  (this Ultralytics version did not expose the confusion matrix; "
              "see runs/test_eval/confusion_matrix.png instead)")
        return None
    return {
        "labels": list(class_names) + ["background"],
        "matrix_predicted_by_true": [[int(v) for v in row] for row in matrix.tolist()],
    }


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

    WEIGHTS_DIR.mkdir(exist_ok=True)
    best_weights = WEIGHTS_DIR / "best.pt"
    if best_weights.exists():
        print(
            f"Found existing weights at {best_weights}, skipping training.\n"
            f"  Delete this file (or the whole weights/ folder) first if you "
            f"changed something and want a fresh retrain."
        )
        finetuned = YOLO(best_weights)
    else:
        finetuned, run_dir = finetune(data_yaml)
        shutil.copy(run_dir / "weights" / "best.pt", best_weights)
        print(f"Committed weights to {best_weights}")

    # ---- Detection metrics on the test split (README Step 5) ---------------
    print("Evaluating on the test split ...")
    test_metrics = finetuned.val(
        data=str(data_yaml), split="test", imgsz=IMG_SIZE, conf=VAL_CONF, device=EVAL_DEVICE,
        project=str(RUNS_DIR), name="test_eval", exist_ok=True,
    )
    overall_metrics = {
        "precision": test_metrics.results_dict.get("metrics/precision(B)"),
        "recall": test_metrics.results_dict.get("metrics/recall(B)"),
        "mAP50": test_metrics.results_dict.get("metrics/mAP50(B)"),
        "mAP50_95": test_metrics.results_dict.get("metrics/mAP50-95(B)"),
    }
    per_class = per_class_metrics(test_metrics, class_names)
    det_confusion = detection_confusion_matrix(test_metrics, class_names)

    # ---- Per-snapshot scores: finetuned model and COCO zero-shot -----------
    print("Scoring the test split with the finetuned model ...")
    rows = score_split(finetuned, class_names, target_class=POSITIVE_CLASS)

    print("Scoring the test split with the pretrained COCO model, zero-shot ...")
    # COCO has no `head` or `helmet` class. Its closest available signal is
    # `person`, so the zero-shot baseline flags a snapshot when it sees a person.
    coco_rows = score_split(YOLO(MODEL_CHECKPOINT), class_names, target_class="person")

    # ---- Cost sweep and the chosen operating point (README Steps 6-7) ------
    print("Sweeping the confidence threshold ...")
    single_threshold_sweep = [three_way_cost(rows, c, c) for c in CONF_GRID]
    best = rank_two_threshold_designs(rows)[0]
    single_best = best_single_threshold(rows)
    everything = human_checks_everything(rows)
    nothing = do_nothing()
    capacity = review_capacity(best["review_share"])
    coco_best = best_single_threshold(coco_rows)

    # ---- Finetuned vs both baselines, same test split (README Step 4) ------
    head_ap50 = next((r["AP50"] for r in per_class if r["class"] == POSITIVE_CLASS), None)
    baselines = [
        {
            "model": f"Finetuned {MODEL_CHECKPOINT}",
            "head_AP50": head_ap50,
            **image_level_metrics(rows, single_best["high"]),
            "cost_per_1000": single_best["cost"],
        },
        {
            "model": "Pretrained COCO, zero-shot ('person' as proxy)",
            "head_AP50": 0.0,  # COCO has no head class: it cannot produce one
            **image_level_metrics(coco_rows, coco_best["high"]),
            "cost_per_1000": coco_best["cost"],
        },
        {
            "model": "Trivial: predict nothing (= majority class 'compliant')",
            "head_AP50": 0.0,
            **image_level_metrics(rows, 1.01),  # no score reaches 1.01: nothing flagged
            "cost_per_1000": nothing["cost"],
        },
    ]

    print("Testing how sensitive the choice is to the assumptions ...")
    sensitivity = sensitivity_table(rows)

    print("\nFinetuned model vs baselines (test split, per snapshot, model deciding alone):")
    for b in baselines:
        print(
            f"  {b['model'][:55]:<55} head AP50 {b['head_AP50'] or 0:.3f} | "
            f"P {b['precision']:.2f} R {b['recall']:.2f} Acc {b['accuracy']:.2f} | "
            f"EUR {b['cost_per_1000']:,.2f} / 1,000"
        )
    print(f"\nChosen operating point: ESCALATE >= {best['low']}, FAIL >= {best['high']}")
    print(f"  EUR {best['cost']:,.2f} / 1,000 snapshots, {best['review_share']:.1%} reviewed")
    print(f"  model-alone best single cut: EUR {single_best['cost']:,.2f} (cut {single_best['high']})")
    print(f"  officer checks everything:   EUR {everything['cost']:,.2f}")
    print(f"  no system at all:            EUR {nothing['cost']:,.2f}")
    print(
        f"  review load: {capacity['reviews_per_day']:.0f} reviews/day, "
        f"needs {capacity['officers_needed']:.2f} safety officer(s)"
    )
    print("\nSensitivity (one assumption moved at a time):")
    for s in sensitivity:
        print(
            f"  {s['assumption']:<19} {s['value']:<9.4g} band [{s['low']}, {s['high']}) "
            f"3-way EUR {s['three_way_cost']:>8,.2f} | alone {s['model_alone_cost']:>8,.2f} | "
            f"all-human {s['officer_checks_everything_cost']:>8,.2f} | "
            f"nothing {s['do_nothing_cost']:>9,.2f}"
        )

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    artifact = {
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION},
        "classes": class_names,
        "positive_class": POSITIVE_CLASS,
        "training": {
            "device": DEVICE, "model_checkpoint": MODEL_CHECKPOINT, "epochs": EPOCHS,
            "imgsz": IMG_SIZE, "patience": PATIENCE, "seed": SEED,
        },
        "model_checkpoint": MODEL_CHECKPOINT,
        "overall_metrics": overall_metrics,
        "per_class_metrics": per_class,
        "detection_confusion_matrix": det_confusion,
        "baselines": baselines,
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
        "sensitivity": sensitivity,
        "test_rows": [row.__dict__ for row in rows],
    }
    (ARTIFACTS_DIR / "results.json").write_text(json.dumps(artifact, indent=2))
    print(f"\nSaved {ARTIFACTS_DIR / 'results.json'} for app.py.")


if __name__ == "__main__":
    main()