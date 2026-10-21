"""
Assignment 2: Gradio dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

from train import (
    CONF_GRID,
    IMG_SIZE,
    POSITIVE_CLASS,
    ScoredImage,
    best_single_threshold,
    confusion_counts,
    do_nothing,
    human_checks_everything,
    review_capacity,
    three_way_cost,
)

ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "weights" / "best.pt"
ARTIFACT_PATH = ROOT / "artifacts" / "results.json"

if not WEIGHTS_PATH.exists():
    raise SystemExit(f"No weights at {WEIGHTS_PATH}. Run `uv run python main.py train` first.")
if not ARTIFACT_PATH.exists():
    raise SystemExit(f"No {ARTIFACT_PATH}. Run `uv run python main.py train` first.")

model = YOLO(str(WEIGHTS_PATH))  # the committed weights, never runs/
ARTIFACT = json.loads(ARTIFACT_PATH.read_text())
ROWS = [ScoredImage(**r) for r in ARTIFACT["test_rows"]]
CHOSEN = ARTIFACT["chosen_operating_point"]

# Fixed context, computed once at startup from the saved test-split scores.
MODEL_ALONE = best_single_threshold(ROWS)
EVERYTHING = human_checks_everything(ROWS)
NOTHING = do_nothing()
SINGLE_CUT_CURVE = [three_way_cost(ROWS, c, c) for c in CONF_GRID]
# Cost as the FAIL edge moves with the ESCALATE edge held at its chosen value:
# the chosen operating point is a point on this curve.
FAIL_EDGE_CURVE = [three_way_cost(ROWS, CHOSEN["low"], c) for c in CONF_GRID if c >= CHOSEN["low"]]


def decide(score: float, low: float, high: float) -> str:
    return "PASS" if score < low else "ESCALATE" if score < high else "FAIL"


def live_check(image, low: float, high: float):
    if image is None:
        return None, "Upload a work-area snapshot to see the model's decision."
    low, high = min(low, high), max(low, high)
    # Draw every box at or above the ESCALATE edge: below it, nothing would happen.
    result = model.predict(image, conf=low, imgsz=IMG_SIZE, verbose=False)[0]
    annotated = result.plot()[:, :, ::-1]  # BGR -> RGB
    head_scores = [
        float(b.conf) for b in result.boxes if model.names[int(b.cls)] == POSITIVE_CLASS
    ]
    score = max(head_scores) if head_scores else 0.0
    lines = [f"- `{model.names[int(b.cls)]}` {float(b.conf):.2f}" for b in result.boxes]
    report = (
        f"### Decision: **{decide(score, low, high)}**\n"
        f"Highest `{POSITIVE_CLASS}` confidence: **{score:.2f}** "
        f"(PASS below {low:.2f}, ESCALATE {low:.2f}-{high:.2f}, FAIL at {high:.2f}+)\n\n"
        + ("\n".join(lines) if lines else "No detections at or above the ESCALATE edge.")
    )
    return annotated, report


def confusion_figure(low: float, high: float) -> go.Figure:
    counts = confusion_counts(ROWS, low, high)
    zones = ["PASS", "ESCALATE", "FAIL"]
    truths = ["violation", "compliant"]
    z = [[counts[t][zone] for zone in zones] for t in truths]
    fig = go.Figure(go.Heatmap(
        z=z, x=zones, y=["True: violation", "True: compliant"],
        text=z, texttemplate="%{text}", colorscale="Blues", showscale=False,
    ))
    fig.update_layout(
        title=f"Test-split snapshots by outcome (n={len(ROWS)})",
        xaxis_title="System decision", margin=dict(t=50, b=40, l=10, r=10), height=320,
    )
    return fig


def cost_curve_figure(low: float, high: float, current_cost: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[d["high"] for d in SINGLE_CUT_CURVE], y=[d["cost"] for d in SINGLE_CUT_CURVE],
        mode="lines", name="Model decides alone (one cut)",
    ))
    fig.add_trace(go.Scatter(
        x=[d["high"] for d in FAIL_EDGE_CURVE], y=[d["cost"] for d in FAIL_EDGE_CURVE],
        mode="lines", name=f"FAIL edge moved, ESCALATE fixed at {CHOSEN['low']}",
    ))
    fig.add_trace(go.Scatter(
        x=[CHOSEN["high"]], y=[CHOSEN["cost"]], mode="markers+text",
        marker=dict(size=13, symbol="star", color="firebrick"),
        text=["chosen"], textposition="top center", name="Chosen operating point",
    ))
    fig.add_trace(go.Scatter(
        x=[high], y=[current_cost], mode="markers",
        marker=dict(size=10, symbol="circle-open", color="black"), name="Current sliders",
    ))
    fig.update_layout(
        title="Expected cost against threshold",
        xaxis_title="Confidence threshold (FAIL edge)",
        yaxis_title="EUR per 1,000 snapshots",
        margin=dict(t=50, b=40, l=10, r=10), height=380,
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


def update_costs(low: float, high: float):
    low, high = min(low, high), max(low, high)
    design = three_way_cost(ROWS, low, high)
    capacity = review_capacity(design["review_share"])
    summary = (
        f"**PASS** below `{low:.2f}` | **ESCALATE** `{low:.2f}`-`{high:.2f}` | "
        f"**FAIL** at `{high:.2f}` and above\n\n"
        f"- Expected cost: **EUR {design['cost']:,.2f}** per 1,000 snapshots\n"
        f"- Sent to the safety officer: **{design['review_share']:.1%}** "
        f"(~{capacity['reviews_per_day']:.0f} a day, "
        f"{capacity['officers_needed']:.2f} of one officer's shift)\n"
        f"- Missed violations: {design['fn']:.1f} per 1,000 | "
        f"false alarms: {design['fp']:.1f} per 1,000"
    )
    comparison = pd.DataFrame([
        {"Design": "Current band", "EUR / 1,000": design["cost"],
         "Reviewed %": design["review_share"] * 100},
        {"Design": "Model decides alone", "EUR / 1,000": MODEL_ALONE["cost"], "Reviewed %": 0.0},
        {"Design": "Officer checks everything", "EUR / 1,000": EVERYTHING["cost"],
         "Reviewed %": 100.0},
        {"Design": "No system", "EUR / 1,000": NOTHING["cost"], "Reviewed %": 0.0},
    ]).round(2)
    return (
        summary,
        confusion_figure(low, high),
        cost_curve_figure(low, high, design["cost"]),
        comparison,
    )


def detection_confusion_figure() -> go.Figure | None:
    cm = ARTIFACT.get("detection_confusion_matrix")
    if not cm:
        return None
    fig = go.Figure(go.Heatmap(
        z=cm["matrix_predicted_by_true"], x=cm["labels"], y=cm["labels"],
        text=cm["matrix_predicted_by_true"], texttemplate="%{text}",
        colorscale="Blues", showscale=False,
    ))
    fig.update_layout(
        title="Detection confusion matrix (test split, boxes)",
        xaxis_title="True class", yaxis_title="Predicted class",
        margin=dict(t=50, b=40, l=10, r=10), height=380,
    )
    return fig


overall = ARTIFACT["overall_metrics"]
dataset = ARTIFACT["dataset"]

with gr.Blocks(title="Hard-hat compliance dashboard") as demo:
    gr.Markdown(
        "# Hard-hat compliance dashboard\n"
        f"Dataset `{dataset['workspace']}/{dataset['project']}` v{dataset['version']} | "
        f"model `{ARTIFACT['model_checkpoint']}` | positive class `{POSITIVE_CLASS}` "
        "(a bare head means a worker without a hard hat)"
    )
    with gr.Row():
        low_slider = gr.Slider(0.01, 0.99, value=CHOSEN["low"], step=0.01,
                               label="ESCALATE edge (below: PASS)")
        high_slider = gr.Slider(0.01, 0.99, value=CHOSEN["high"], step=0.01,
                                label="FAIL edge (at or above: FAIL)")
    gr.Markdown(
        f"Chosen operating point from the sweep: ESCALATE at **{CHOSEN['low']}**, "
        f"FAIL at **{CHOSEN['high']}**."
    )

    with gr.Tab("Live check"):
        with gr.Row():
            image_in = gr.Image(type="pil", label="Work-area snapshot")
            image_out = gr.Image(label="Detections")
        decision_md = gr.Markdown()
        for trigger in (image_in, low_slider, high_slider):
            trigger.change(live_check, [image_in, low_slider, high_slider],
                           [image_out, decision_md])

    with gr.Tab("Threshold & cost"):
        summary_md = gr.Markdown()
        with gr.Row():
            outcome_plot = gr.Plot()
            cost_plot = gr.Plot()
        comparison_table = gr.Dataframe(label="Designs compared (EUR per 1,000 snapshots)")
        outputs = [summary_md, outcome_plot, cost_plot, comparison_table]
        for slider in (low_slider, high_slider):
            slider.change(update_costs, [low_slider, high_slider], outputs)
        demo.load(update_costs, [low_slider, high_slider], outputs)

    with gr.Tab("Per-class performance"):
        gr.Markdown(
            f"Overall on the test split: mAP50 **{overall['mAP50']:.3f}**, "
            f"mAP50-95 **{overall['mAP50_95']:.3f}**, precision **{overall['precision']:.3f}**, "
            f"recall **{overall['recall']:.3f}**. The overall number hides a weak class: "
            "check the table."
        )
        gr.Dataframe(pd.DataFrame(ARTIFACT["per_class_metrics"]).round(3),
                     label="Per class: precision, recall, AP50, AP50-95")
        det_fig = detection_confusion_figure()
        if det_fig is not None:
            gr.Plot(det_fig)
        if isinstance(ARTIFACT.get("baselines"), list):
            gr.Dataframe(
                pd.DataFrame(ARTIFACT["baselines"])[
                    ["model", "head_AP50", "precision", "recall", "accuracy", "cost_per_1000"]
                ].round(3),
                label="Finetuned model vs baselines (test split, per snapshot, model alone)",
            )

if __name__ == "__main__":
    demo.launch()