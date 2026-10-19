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
    raise SystemExit(
        f"No weights at {WEIGHTS_PATH}. Run `uv run python main.py train` first, "
        f"or make sure weights/best.pt is committed."
    )
if not ARTIFACT_PATH.exists():
    raise SystemExit(
        f"No {ARTIFACT_PATH}. This dashboard reads train.py's saved results "
        f"instead of recomputing them; run `uv run python main.py train` first."
    )
 
model = YOLO(str(WEIGHTS_PATH))  # loads the committed weights only, never runs/
CLASS_NAMES = model.names
 
ARTIFACT = json.loads(ARTIFACT_PATH.read_text())
ROWS = [ScoredImage(**r) for r in ARTIFACT["test_rows"]]
PER_CLASS = ARTIFACT["per_class_metrics"]
OVERALL = ARTIFACT["overall_metrics"]
CHOSEN = ARTIFACT["chosen_operating_point"]
COST_SWEEP = ARTIFACT["cost_sweep_single_threshold"]
 
 
def predict_image(image, conf: float):
    if image is None:
        return None, "Upload a snapshot to see the model's predictions."
    result = model.predict(image, conf=conf, verbose=False)[0]
    annotated = result.plot()[:, :, ::-1]  # BGR -> RGB
    lines = [f"- `{CLASS_NAMES[int(b.cls)]}`  {float(b.conf):.2f}" for b in result.boxes]
    report = "\n".join(lines) if lines else "No detections above this threshold."
    return annotated, report
 
 
def make_confusion_figure(low: float, high: float) -> go.Figure:
    counts = confusion_counts(ROWS, low, high)
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=["", "PASS", "ESCALATE", "FAIL"], align="left"),
                cells=dict(
                    values=[
                        ["Ground truth: violation", "Ground truth: compliant"],
                        [counts["violation"]["PASS"], counts["compliant"]["PASS"]],
                        [counts["violation"]["ESCALATE"], counts["compliant"]["ESCALATE"]],
                        [counts["violation"]["FAIL"], counts["compliant"]["FAIL"]],
                    ],
                    align="left",
                ),
            )
        ]
    )
    fig.update_layout(title="Test-split snapshots landing in each outcome", margin=dict(t=40, b=10))
    return fig
 
 
def make_cost_curve_figure(low: float, high: float) -> go.Figure:
    xs = [row["high"] for row in COST_SWEEP]
    ys = [row["cost"] for row in COST_SWEEP]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="model decides alone (single cut)"))
    fig.add_vline(x=CHOSEN["high"], line_dash="dash", line_color="firebrick",
                  annotation_text="chosen FAIL cut")
    fig.add_vline(x=CHOSEN["low"], line_dash="dot", line_color="darkorange",
                  annotation_text="chosen ESCALATE cut")
    fig.add_vline(x=high, line_color="firebrick", opacity=0.4)
    fig.add_vline(x=low, line_color="darkorange", opacity=0.4)
    fig.update_layout(
        title="Cost against threshold (single-cut baseline)",
        xaxis_title="confidence threshold",
        yaxis_title="EUR / 1,000 snapshots",
        margin=dict(t=40, b=10),
    )
    return fig
 
 
def update_dashboard(low: float, high: float):
    if low > high:
        low, high = high, low
    design = three_way_cost(ROWS, low, high)
    single = best_single_threshold(ROWS)
    everything = human_checks_everything(ROWS)
    nothing = do_nothing()
    capacity = review_capacity(design["review_share"])
 
    comparison_df = pd.DataFrame(
        [
            {"Design": "Chosen 3-way band", "EUR / 1,000": design["cost"], "Reviewed %": design["review_share"] * 100},
            {"Design": "Model decides alone", "EUR / 1,000": single["cost"], "Reviewed %": 0.0},
            {"Design": "Officer checks everything", "EUR / 1,000": everything["cost"], "Reviewed %": 100.0},
            {"Design": "No system (do nothing)", "EUR / 1,000": nothing["cost"], "Reviewed %": 0.0},
        ]
    )
 
    summary = (
        f"**PASS** below `{low:.2f}` &nbsp;|&nbsp; **ESCALATE** `{low:.2f}`-`{high:.2f}` "
        f"&nbsp;|&nbsp; **FAIL** at/above `{high:.2f}`\n\n"
        f"- Expected cost: **EUR {design['cost']:,.2f}** per 1,000 snapshots\n"
        f"- Sent to the safety officer: **{design['review_share']:.1%}**\n"
        f"- Missed violations (FN): {design['fn']:.2f} / 1,000\n"
        f"- False alarms (FP): {design['fp']:.2f} / 1,000\n"
        f"- Review load: ~{capacity['reviews_per_day']:.0f} reviews/day, "
        f"needs **{capacity['officers_needed']:.2f}** safety officer(s)"
    )
    return make_confusion_figure(low, high), make_cost_curve_figure(low, high), comparison_df, summary
 
 
with gr.Blocks(title="Hard-Hat Compliance Dashboard") as demo:
    gr.Markdown("# Hard-Hat Compliance Dashboard")
    gr.Markdown(
        f"Dataset: `{ARTIFACT['dataset']['workspace']}/{ARTIFACT['dataset']['project']}` "
        f"v{ARTIFACT['dataset']['version']}  |  Model: `{ARTIFACT['model_checkpoint']}`  |  "
        f"Positive class: `{ARTIFACT['positive_class']}`"
    )
 
    with gr.Tab("Live check"):
        with gr.Row():
            image_in = gr.Image(type="pil", label="Work-area snapshot")
            image_out = gr.Image(label="Detections")
        conf_slider = gr.Slider(0.01, 0.99, value=CHOSEN["high"], step=0.01, label="Confidence threshold")
        report = gr.Markdown()
        image_in.change(predict_image, [image_in, conf_slider], [image_out, report])
        conf_slider.change(predict_image, [image_in, conf_slider], [image_out, report])
 
    with gr.Tab("Threshold & cost"):
        with gr.Row():
            low_slider = gr.Slider(0.01, 0.99, value=CHOSEN["low"], step=0.01, label="ESCALATE starts at (low)")
            high_slider = gr.Slider(0.01, 0.99, value=CHOSEN["high"], step=0.01, label="FAIL starts at (high)")
        summary_md = gr.Markdown()
        with gr.Row():
            cm_plot = gr.Plot(label="Outcome counts")
            cost_plot = gr.Plot(label="Cost vs. threshold")
        comparison_table = gr.Dataframe(label="Design comparison (EUR per 1,000 snapshots)")
 
        for slider in (low_slider, high_slider):
            slider.change(
                update_dashboard, [low_slider, high_slider],
                [cm_plot, cost_plot, comparison_table, summary_md],
            )
        demo.load(
            update_dashboard, [low_slider, high_slider],
            [cm_plot, cost_plot, comparison_table, summary_md],
        )
 
    with gr.Tab("Per-class performance"):
        gr.Markdown(
            f"Overall: mAP50 **{OVERALL['mAP50']:.3f}**, mAP50-95 **{OVERALL['mAP50_95']:.3f}**, "
            f"precision **{OVERALL['precision']:.3f}**, recall **{OVERALL['recall']:.3f}**"
        )
        gr.Dataframe(pd.DataFrame(PER_CLASS), label="Per-class precision / recall / AP50 / AP50-95")
 
if __name__ == "__main__":
    demo.launch()