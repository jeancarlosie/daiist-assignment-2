# Assignment 2 Report

- **Name**: Jean Carlos
- **Student ID**: 19832
- **Email**: jcarlos.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Business problem, and the cost of being wrong

The model supports 1 decision: does a site-camera snapshot show a worker without a hard hat? 5 fixed cameras each take a snapshot every 2 minutes over a 10-hour shift (1,500 snapshots a day). The positive class is a violation: a snapshot with at least 1 bare head. For each snapshot, the system does 1 of 3 things:

1. PASS: nothing happens.
2. ESCALATE: the site safety officer checks it on their phone (~45 seconds).
3. FAIL: an alarm sounds and a supervisor stops the task.

The 2 mistakes are not equally bad:

- A miss (FALSE NEGATIVE) leaves a bare-headed worker unprotected until the next snapshot, risking a head injury. Cost: €50 per miss.
- A false alarm (FALSE POSITIVE) stops a compliant worker while a supervisor clears it. Cost: €2.17.

A miss is ~23x worse, which is why both thresholds are low, why I care about RECALL on violations more than precision, and why I ignore accuracy (always predicting "compliant" scores 80–90% while catching nothing).

## Dataset

Roboflow Universe: workspace "joseph-nelson", project "hard-hat-workers", version 10.
Link: https://universe.roboflow.com/joseph-nelson/hard-hat-workers/dataset/10

Each image is a photo of 1 or more people, mostly on or around construction sites. The classes are "head" (a head without a hard hat), "helmet" (a head with a hard hat) and "person" (a whole person, not used in the decision).

The dataset serves the problem rather than just resembling it, because its labels mark exactly what the decision needs: whether each individual head is bare or protected. A snapshot with at least 1 "head" is a violation, so the model's output maps straight onto PASS, ESCALATE or FAIL.

## What the dataset's composition implies

| Split | Images | `head` boxes (images) | `helmet` boxes (images) | `person` boxes (images) |
|---|---|---|---|---|
| Train | 4,916 | 4,612 (911) | 13,919 (4,503) | 450 (154) |
| Valid | 1,413 | 1,339 (256) | 3,913 (1,297) | 101 (32) |
| Test | 706 | 726 (140) | 1,915 (636) | 64 (23) |

The split is 70 / 20 / 10. The median box covers ~0.6% of the image for "head", ~0.8% for "helmet" and 6–8% for "person".

This has 4 consequences:

1. "person" is too thin to trust: 450 training and 64 test boxes (23 images), against 4,612 / 726 for "head" and 13,919 / 1,915 for "helmet". The vast majority of people in the photos are not labelled "person", so the model barely learned the class, and its test results rest on too few examples to mean much.
2. `head` is usable but imprecise: with 140 violation SNAPSHOTS in the test split, a recall of 0.95 has a 95% confidence interval of 0.90–0.98.
3. Heads are tiny, so distant heads are hard to detect.
4. The test split looks like the training photos (web and press photos, often posed and close up), not like a fixed site camera. It also contains 19.8% violations, against the 10% I assume on site, so I re-weight all costs to 10%.

So I can't conclude that the recall measured here would hold on a real site, or anything about "person".

## Finetuning and baselines

I used DETECTION, not classification, because the decision depends on finding each individual head, which is often small in a crowded photo. A snapshot's score is its highest "head" confidence (0 if there is none).

The model is yolo11n, fine-tuned from the COCO-pretrained checkpoint. I trained it on an Apple M1 laptop in ~5 hours, with 25 epochs, image size 416, batch 8, mosaic augmentation off and seed 0. These settings are smaller than usual only because of my laptop: yolo11s at image size 640 couldn't finish a single training batch in 7 minutes. Validation mAP50 had flattened by ~epoch 17, so 25 epochs was more than enough.

Results on the same test split (706 snapshots; each model decides alone, at its own cheapest threshold):

| Model | `head` AP50 | Recall | Precision | Accuracy | € per 1,000 snapshots |
|---|---|---|---|---|---|
| Fine-tuned yolo11n | 0.955 | 0.95 | 0.79 | 0.94 | 371 |
| Pretrained COCO, zero-shot | 0 | 0.99 | 0.23 | 0.32 | 1,679 |
| Trivial: always "compliant" | 0 | 0 | – | 0.80 | 5,000 |

Overall, the fine-tuned model scores mAP50 0.652 and mAP50-95 0.448. COCO has no "head" class, so its closest zero-shot signal is "person": it flags nearly EVERY snapshot with a person in it, giving high recall but very low precision. At the site's 10% violation rate, the fine-tuned model's precision would be 0.63 rather than 0.79.

Fine-tuning cut the expected cost ~13x compared with the trivial baseline and ~4.5x compared with COCO. The gap was larger than I expected on cost, but accuracy hides it (it only rises from 0.80 to 0.94), and overall mAP50 is lower than I expected because of "person".

## Per-class evaluation and failure modes

| Class | Precision | Recall | AP50 | AP50-95 |
|---|---|---|---|---|
| `head` | 0.930 | 0.902 | 0.955 | 0.659 |
| `helmet` | 0.945 | 0.922 | 0.968 | 0.667 |
| `person` | 1.000 | 0.000 | 0.033 | 0.019 |

"helmet" and "head" carry the headline number; "person" drags it down to 0.652. "person" shows precision 1 and recall 0 because no "person" box survives at a usable confidence: most people in the photos are unlabelled, so the class is unreliable to train and to score.

Detection confusion matrix on the test split, at confidence 0.25 (rows are predictions, columns the true class):

| Predicted ↓ / True → | head | helmet | person | background |
|---|---|---|---|---|
| head | 678 | 2 | 0 | 104 |
| helmet | 20 | 1,817 | 0 | 201 |
| person | 0 | 0 | 0 | 0 |
| background (missed) | 28 | 96 | 64 | – |

The main confusions:

- 20 bare heads read as `helmet`: the EXPENSIVE error, a missed violation.
- 2 helmets read as `head`: the cheap error, a false alarm.
- 28 bare heads missed ENTIRELY.
- 104 `head` boxes drawn on nothing, causing false alarms.

It breaks in 3 places:

1. Distant heads: 6 of the 140 test violations score 0.00, passing every threshold. Most are wide shots with small, distant heads.
2. LABEL ERRORS: the compliant snapshots it scores highest (up to 0.84) include clear bare heads, such as a man holding his helmet instead of wearing it. Here the model is right and the label is wrong, so TRUE SPECIFICITY is likely better than measured.
3. Unfamiliar photos: in a party photo, hats and bare heads were read as `helmet`, so it passed despite 3 violations.

Because some compliant snapshots score up to 0.84 and some violations score 0.00, NO THRESHOLD can separate every case.

## Cost matrix and operating point

| Outcome | Cost | Where the figure comes from |
|---|---|---|
| Missed violation (FN) | €50 | OSHA's Safety Pays estimator puts an average fracture at $115,197 including indirect costs (https://www.osha.gov/safetypays/estimator-text). I round this to €100,000 as a proxy for a head injury, then multiply by an assumed 1-in-2,000 chance that a missed snapshot leads to an injury (my judgment call). |
| False alarm (FP) | €2.17 | 4 minutes of a supervisor at €25/hour + 30% employer costs. |
| Review (ESCALATE) | €0.36 | 45 seconds of a safety officer at €22/hour + 30% employer costs. |
| Correct decision (TP, TN) | €0 | Stopping an unsafe task is the purpose of the alarm, not a cost. |

I treat the officer as FALLIBLE: they catch 97% of escalated violations and wrongly confirm 5% of compliant snapshots.

I assume a base rate of 10% (1 snapshot in 10 shows a violation). In a study of 18 Irish construction sites, 7 sites had 5% or fewer workers without a hard hat and 8 had 20–70% (eLCOSH, https://elcosh.org/document/1435/693/d000055/results.html). I chose 10%, closer to the well-run sites.

I priced every pair of thresholds from 0.02 to 0.98 (in steps of 0.02) on the test split, re-weighted to the 10% base rate. The cost-against-threshold curve is in the dashboard's Threshold & cost tab, with the chosen point marked.

- Model deciding alone: the cheapest single threshold is 0.06, at €371 per 1,000 snapshots.
- Chosen operating point: ESCALATE from 0.04, FAIL from 0.52, at €284 per 1,000.

The thresholds are low because misses cost 23x more than false alarms, and most compliant snapshots score below 0.1, so even weak "head" detections are worth a look.

The assumption that would most change the answer is the BASE RATE: moving it from 2% to 40% moves the FAIL threshold from 0.72 down to 0.12. Across EVERYTHING I varied (injury probability, base rate, review cost and officer recall), the three-way design stayed the cheapest. It would only lose to the model deciding alone if a review cost about €1.43-2.86 (4–8x my estimate).

## What a person still has to check

With ESCALATE from 0.04 and FAIL from 0.52, 7.4% of snapshots go to a person (~111 a day). At 45 seconds each, that is 1.4 hours a day, or 14% of one safety officer's shift, which is manageable. The catch is that ~86% of escalated snapshots are compliant, so the officer may stop paying attention. Mixing a few known violations into the queue would keep them alert and measure their real accuracy.

Priced against the alternatives (per 1,000 snapshots):

| Design | € per 1,000 | Sent to a person | Missed violations | False alarms |
|---|---|---|---|---|
| Three-way band (chosen) | 284 | 7.4% (0.14 officers) | 4.6 | 12.7 |
| Model decides alone | 371 | 0% | 5.0 | 56 |
| Person checks everything | 605 | 100% (1.9 officers) | 3.0 | 45 |
| Do nothing | 5,000 | 0% | 100 | 0 |

The band beats the model alone because an uncertain case costs a €0.36 review instead of a €2.17 false alarm. It beats checking everything because the officer only sees the 7.4% of snapshots the model is unsure about.

## Limitations & next steps

1. The test photos are NOT site-camera images. I would label a few hundred frames from the real cameras, measure recall on them, and fine-tune on them.
2. The LABELS are imperfect: some "compliant" images show bare heads, and the vast majority of people are not labelled "person". I would relabel the worst-scoring cases shown in the dashboard, and drop or relabel "person".
3. Distant heads are missed (6 of 140 violations score 0.00). I would retrain on a GPU with yolo11s at image size 640, or split each frame into tiles.
4. The costs rest on ASSUMPTIONS: the 1-in-2,000 injury chance, a fracture standing in for a head injury, and 97% officer recall. I would use figures from a Spanish occupational insurer (mutua) or the site's own incident records, and measure officer recall with known violations in the queue.
5. Snapshots are treated as independent, but the same worker appears in consecutive ones, so the cost of a single miss may be OVERSTATED. I would track workers across frames.

## Generative AI use disclosure

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*

## Generative AI use disclosure

I used Claude (Anthropic's AI assistant, via claude.ai) throughout this assignment:

I used Claude to write most of train.py (the training pipeline, cost model and threshold sweep) and the entirety of app.py (the Gradio dashboard), and to help me debug them, including the slow training and evaluation on my M1 laptop. I ran every stage myself, checked the outputs, and tested the dashboard with my own images. In terms of business framing, I used Claude to help me research the typical cost figures and find sources (OSHA Safety Pays, eLCOSH) to justify my final choices, but I chose the dataset, brainstormed the three-way design, reviewed all the assumptions, and wrote the Report myself (using Claude to reword certain sections for improved readability).