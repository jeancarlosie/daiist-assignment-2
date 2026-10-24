"""
Assignment 2: training pipeline (run via `uv run python main.py train`)
"""

from __future__ import annotations

import json
import os
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



# PART 1 - cost / decision model

# 1a. Dataset and training configuration

WORKSPACE = "joseph-nelson"
PROJECT = "hard-hat-workers"
VERSION = 10  # 70/20/10 split; classes head, helmet, person
ROBOFLOW_FORMAT = "yolov8"  # label layout Ultralytics reads; YOLO11 uses the same one

POSITIVE_CLASS = "head"  # the dataset's label for a bare head

MODEL_CHECKPOINT = "yolo11n.pt"  # nano: trainable on an M1 laptop; yolo11s needs a GPU
EPOCHS = 25  # validation mAP50 had flattened by epoch 17
IMG_SIZE = 416  # not 640: compute scales with imgsz^2
BATCH = 16
PATIENCE = 15  # stop early if validation mAP stops improving
SEED = 0

# Confidence floor for evaluation and scoring; Ultralytics' near-zero default
# floods non-maximum suppression with junk boxes, which stalled for hours on
# MPS
VAL_CONF = 0.01

# Confusion matrix uses the usual 0.25: at 0.01 it is swamped by junk boxes
CM_CONF = 0.25

SNAPSHOT_INTERVAL_SECONDS = 120  # one snapshot per camera every two minutes


# 1b. The cost matrix (EUR)

# FN (missed violation) = P(injury in that 2-minute window) x injury cost
# Injury cost: OSHA Safety Pays, fracture: USD 54,856 direct x (1 + 1.1
# indirect ratio) = USD 115,197, rounded to EUR 100,000 (NCCI claims 2015-17)
AVG_INJURY_COST_EUR = 100_000.00
P_INJURY_IF_MISSED = 1 / 2000  # judgment call; swept in sensitivity_table()

COST_FN = AVG_INJURY_COST_EUR * P_INJURY_IF_MISSED  # EUR 50.00

LOADED_LABOUR_MULTIPLIER = 1.3  # wage + 30% employer costs

# FP (alarm on a compliant worker): a supervisor's time to clear it
SUPERVISOR_RATE_EUR_PER_HOUR = 25.00
FALSE_ALARM_MINUTES = 4
COST_FP = SUPERVISOR_RATE_EUR_PER_HOUR * LOADED_LABOUR_MULTIPLIER / 60 * FALSE_ALARM_MINUTES

# A review (ESCALATE): the safety officer checks the snapshot
REVIEW_SECONDS = 45
SAFETY_OFFICER_RATE_EUR_PER_HOUR = 22.00
COST_REVIEW = SAFETY_OFFICER_RATE_EUR_PER_HOUR * LOADED_LABOUR_MULTIPLIER / 3600 * REVIEW_SECONDS

# The officer is NOT an oracle: an escalated violation can be waved through
# and a compliant snapshot wrongly confirmed; recall is swept in sensitivity_table()
SAFETY_OFFICER_RECALL = 0.97
SAFETY_OFFICER_SPECIFICITY = 0.95

# Judgment call: in a study of 18 Irish sites, 7 sites had 5% or fewer of workers
# outdoors without a hard hat and 8 had 20-70%; swept in sensitivity_table()
BASE_RATE = 0.10


# 1c. Review capacity: how many people the chosen band needs

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



# 1d. Threshold grid and per-snapshot scores

CONF_GRID: tuple[float, ...] = tuple(round(0.02 * i, 2) for i in range(1, 50))  # 0.02 .. 0.98


@dataclass
class ScoredImage:
    image: str
    truth: str
    score: float


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
    """Expected cost per `n_units` snapshots for the band [low, high).
    Zone rates are measured separately per true class, then re-projected
    onto the base rate."""
    if low > high:
        low, high = high, low
    violations = [r for r in rows if r.truth == "violation"]
    compliant = [r for r in rows if r.truth == "compliant"]

    v_pass, v_review, _v_fail = zone_rates(violations, low, high)
    _c_pass, c_review, c_fail = zone_rates(compliant, low, high)

    n_violation = n_units * base_rate
    n_compliant = n_units * (1 - base_rate)

    # A missed violation: auto-passed or escalated and waved through by the officer
    fn = n_violation * (v_pass + v_review * (1 - officer_recall))
    # A false alarm: auto-failed, or escalated and the officer wrongly confirms it
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
    designs = [
        three_way_cost(rows, low, high, **kwargs)
        for i, low in enumerate(grid)
        for high in grid[i:]
    ]
    return sorted(designs, key=lambda d: d["cost"])


def best_single_threshold(
    rows: list[ScoredImage], grid: tuple[float, ...] = CONF_GRID, **kwargs
) -> dict:
    designs = [three_way_cost(rows, c, c, **kwargs) for c in grid]
    return min(designs, key=lambda d: d["cost"])


def human_checks_everything(rows: list[ScoredImage], **kwargs) -> dict:
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
    """Raw counts of test-split snapshots in each zone (not re-projected)."""
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


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson confidence interval for a proportion."""
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    centre = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    half = z * ((p * (1 - p) / trials + z * z / (4 * trials * trials)) ** 0.5) / (1 + z * z / trials)
    return max(0.0, centre - half), min(1.0, centre + half)


def image_level_metrics(
    rows: list[ScoredImage], cut: float, base_rate: float = BASE_RATE
) -> dict:
    """Snapshot-level metrics when every score >= cut is flagged. Precision
    and accuracy depend on the class mix, so both are also re-projected onto
    the site's base rate."""
    tp = sum(1 for r in rows if r.truth == "violation" and r.score >= cut)
    fp = sum(1 for r in rows if r.truth == "compliant" and r.score >= cut)
    fn = sum(1 for r in rows if r.truth == "violation" and r.score < cut)
    tn = sum(1 for r in rows if r.truth == "compliant" and r.score < cut)
    n = len(rows) or 1
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    flagged_at_site = base_rate * recall + (1 - base_rate) * (1 - specificity)
    return {
        "cut": cut, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": recall,
        "recall_95ci": wilson_interval(tp, tp + fn),
        "specificity": specificity,
        "specificity_95ci": wilson_interval(tn, tn + fp),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "accuracy": (tp + tn) / n,
        "precision_at_site": base_rate * recall / flagged_at_site if flagged_at_site else 0.0,
        "accuracy_at_site": base_rate * recall + (1 - base_rate) * specificity,
    }


def sensitivity_table(rows: list[ScoredImage]) -> list[dict]:
    """Re-choose the band while moving one assumption at a time."""
    scenarios = [
        ("p_injury_if_missed", P_INJURY_IF_MISSED * m, {"cost_fn": COST_FN * m})
        for m in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
    ] + [
        ("base_rate", b, {"base_rate": b})
        for b in (0.02, 0.05, 0.10, 0.20, 0.40)
    ] + [
        ("cost_review_eur", COST_REVIEW * m, {"cost_review": COST_REVIEW * m})
        for m in (0.5, 1.0, 2.0, 4.0, 8.0)
    ] + [
        ("officer_recall", r, {"officer_recall": r})
        for r in (0.99, 0.97, 0.90, 0.80, 0.60)
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



# PART 2 -  training pipeline


if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Post-training evaluation stalled for hours on MPS, so it runs on CPU
EVAL_DEVICE = "cpu" if DEVICE == "mps" else DEVICE


def _ensure_absolute_path_key(data_yaml: Path, dataset_dir: Path) -> None:
    """Roboflow's data.yaml uses paths relative to itself, but Ultralytics
    resolves them from the working directory. An absolute `path:` fixes that."""
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
    # On MPS, Ultralytics loads images on a single thread and mosaic made that
    # the bottleneck, so MPS/CPU use batch 8 and NO mosaic
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
        cache="disk",  # decode each image once, not every epoch
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


def dataset_summary(class_names: list[str]) -> dict:
    """Images per split, boxes and images per class, and box size as a share
    of the image, computed from the label files."""
    summary = {}
    for split in ("train", "valid", "test"):
        labels_dir = DATASET_DIR / split / "labels"
        images_dir = DATASET_DIR / split / "images"
        n_images = sum(
            1 for p in images_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        per_class = {name: {"instances": 0, "images": 0, "areas": []} for name in class_names}
        empty = 0
        for label_path in labels_dir.glob("*.txt"):
            lines = [l.split() for l in label_path.read_text().splitlines() if l.strip()]
            if not lines:
                empty += 1
            seen = set()
            for parts in lines:
                name = class_names[int(parts[0])]
                per_class[name]["instances"] += 1
                per_class[name]["areas"].append(float(parts[3]) * float(parts[4]))
                seen.add(name)
            for name in seen:
                per_class[name]["images"] += 1
        classes = {}
        for name, stats in per_class.items():
            areas = sorted(stats["areas"])
            classes[name] = {
                "instances": stats["instances"],
                "images": stats["images"],
                "mean_box_area_pct": 100 * sum(areas) / len(areas) if areas else 0.0,
                "median_box_area_pct": 100 * areas[len(areas) // 2] if areas else 0.0,
            }
        summary[split] = {"images": n_images, "images_without_labels": empty, "classes": classes}
    return summary


def failure_cases(rows: list[ScoredImage], low: float, n: int = 6) -> dict:
    """The highest-scoring compliant snapshots and lowest-scoring violations,
    copied into artifacts/ so the dashboard can show them without the dataset."""
    out_dir = ARTIFACTS_DIR / "failures"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    compliant = sorted((r for r in rows if r.truth == "compliant"), key=lambda r: -r.score)
    violations = sorted((r for r in rows if r.truth == "violation"), key=lambda r: r.score)

    def keep(selection, kind):
        kept = []
        for i, r in enumerate(selection[:n]):
            target = out_dir / f"{kind}_{i}{Path(r.image).suffix}"
            shutil.copy(ROOT / r.image, target)
            kept.append({"image": str(target.relative_to(ROOT)), "score": r.score})
        return kept

    return {
        "confident_false_alarms": keep(compliant, "false_alarm"),
        "weakest_violations": keep(violations, "missed"),
        "highest_score_on_compliant": compliant[0].score if compliant else 0.0,
        "lowest_score_on_violation": violations[0].score if violations else 0.0,
        "violations_below_escalate_edge": sum(1 for r in violations if r.score < low),
    }


def score_split(
    model: YOLO,
    class_names: list[str],
    target_class: str = POSITIVE_CLASS,
    split: str = "test",
    chunk_size: int = 32,
) -> list[ScoredImage]:
    """One row per test image: its true label and the model's highest
    confidence for `target_class` ("head" for the finetuned model, "person"
    for the COCO baseline). Predicts in chunks of images to save time."""
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
    """Per-class precision, recall, AP50 and AP50-95."""
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


def detection_confusion_matrix(model: YOLO, data_yaml: Path, class_names: list[str]) -> dict | None:
    """Box-level confusion matrix on the test split at CM_CONF, stored as
    matrix[predicted, true]. Returns None if it comes back empty."""
    metrics = model.val(
        data=str(data_yaml), split="test", imgsz=IMG_SIZE, conf=CM_CONF,
        device=EVAL_DEVICE, verbose=False,
        plots=True,  # Ultralytics only fills the confusion matrix when plotting
        project=str(RUNS_DIR), name="test_eval_cm", exist_ok=True,
    )
    matrix = getattr(getattr(metrics, "confusion_matrix", None), "matrix", None)
    if matrix is None or float(matrix.sum()) == 0:
        print("  WARNING: the confusion matrix came back empty; it will not be shown.")
        return None
    return {
        "conf": CM_CONF,
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

    print("Describing the dataset from its label files ...")
    data_summary = dataset_summary(class_names)
    for split, info in data_summary.items():
        per_class = ", ".join(
            f"{name} {c['instances']} boxes in {c['images']} images "
            f"(median box {c['median_box_area_pct']:.2f}% of image)"
            for name, c in info["classes"].items()
        )
        print(f"  {split:<5} {info['images']} images: {per_class}")

    WEIGHTS_DIR.mkdir(exist_ok=True)
    best_weights = WEIGHTS_DIR / "best.pt"
    if os.environ.get("REUSE_WEIGHTS") == "1" and best_weights.exists():
        print(f"REUSE_WEIGHTS=1: evaluating the committed {best_weights}, skipping training.")
        finetuned = YOLO(best_weights)
    else:
        finetuned, run_dir = finetune(data_yaml)
        shutil.copy(run_dir / "weights" / "best.pt", best_weights)
        print(f"Committed weights to {best_weights}")

    # Detection metrics on the test split
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
    print(f"Building the detection confusion matrix at conf={CM_CONF} ...")
    det_confusion = detection_confusion_matrix(finetuned, data_yaml, class_names)

    # Per-snapshot scores: finetuned model and COCO zero-shot
    print("Scoring the test split with the finetuned model ...")
    rows = score_split(finetuned, class_names, target_class=POSITIVE_CLASS)

    print("Scoring the test split with the pretrained COCO model, zero-shot ...")
    # COCO has no head or helmet class; `person` is its closest signal
    coco_rows = score_split(YOLO(MODEL_CHECKPOINT), class_names, target_class="person")

    # Cost sweep and the chosen operating point
    print("Sweeping the confidence threshold ...")
    single_threshold_sweep = [three_way_cost(rows, c, c) for c in CONF_GRID]
    best = rank_two_threshold_designs(rows)[0]
    single_best = best_single_threshold(rows)
    everything = human_checks_everything(rows)
    nothing = do_nothing()
    capacity = review_capacity(best["review_share"])
    coco_best = best_single_threshold(coco_rows)

    # Finetuned vs both baselines, same test split
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

    print("Collecting the failure cases ...")
    failures = failure_cases(rows, best["low"])

    print("\nFinetuned model vs baselines (test split, per snapshot, model deciding alone):")
    for b in baselines:
        print(
            f"  {b['model'][:55]:<55} head AP50 {b['head_AP50'] or 0:.3f} | "
            f"R {b['recall']:.2f} Spec {b['specificity']:.2f} P {b['precision']:.2f} "
            f"(P at {BASE_RATE:.0%} base rate {b['precision_at_site']:.2f}) "
            f"Acc {b['accuracy']:.2f} | EUR {b['cost_per_1000']:,.2f} / 1,000"
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
    lo, hi = baselines[0]["recall_95ci"]
    print(
        f"\nModel-alone recall {baselines[0]['recall']:.3f}, 95% interval "
        f"{lo:.3f}-{hi:.3f} on {baselines[0]['tp'] + baselines[0]['fn']} violation snapshots"
    )
    print(
        f"Highest score on a compliant snapshot {failures['highest_score_on_compliant']:.3f}, "
        f"lowest on a violation {failures['lowest_score_on_violation']:.3f}; "
        f"{failures['violations_below_escalate_edge']} violations pass below the ESCALATE edge"
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
        "dataset_summary": data_summary,
        "per_class_metrics": per_class,
        "detection_confusion_matrix": det_confusion,
        "failure_cases": failures,
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