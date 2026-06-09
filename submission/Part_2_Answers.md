# Part 2: Reasoning Questions

## Q1: When should you choose classification vs object detection vs instance segmentation for a computer vision task?

The choice comes down to the granularity of information your downstream system needs, and what you're willing to pay in annotation cost and inference latency.

**Classification** is appropriate when your decision is image-level — quality control pass/fail, content moderation flags, or medical triage screening. It's a single forward pass, cheap to annotate, and scales well. The limitation is obvious: you get no spatial information.

**Object detection** is the workhorse for most production systems that need to count, localize, or track objects. Bounding boxes strike a practical balance between information density and annotation effort. For this glove project, detection was the right call because compliance logic requires knowing *which* hands are bare and *where* they are in the frame — a classification model saying "bare hand somewhere in this image" isn't actionable for a safety officer reviewing footage.

**Instance segmentation** becomes necessary when pixel-accurate boundaries matter: robot grasp planning, surgical instrument tracking, or precise area measurements. The cost is substantial — polygon annotation is 5-10x slower than boxes, and mask heads (like Mask R-CNN's or YOLACT's) add meaningful compute overhead. I generally avoid segmentation unless the downstream task explicitly requires it.

A useful heuristic: start with detection. You can always add a lightweight segmentation head later if you discover your bounding box overlap is too high for your post-processing logic.

## Q2: You trained a model for 50 epochs and notice the validation loss stopped improving after epoch 30, while training loss keeps dropping. What is happening and how do you fix it?

This is textbook overfitting. The model is fitting noise in the training distribution — learning spurious correlations like background textures or lighting conditions specific to your training set — rather than the invariant features that generalize. The divergence between training and validation loss curves is the clearest diagnostic signal.

My typical remediation stack, in order of effort:

1. **Early stopping** — the first and cheapest fix. I usually set patience to 5-10 epochs with a delta threshold of 0.001 on validation mAP or loss. There's rarely value in training past the inflection point.

2. **Augmentation hardening** — the default YOLO augmentations are a baseline, not a ceiling. I increase mosaic probability, add random perspective distortion, and sometimes implement copy-paste augmentation for underrepresented classes. The goal is to make the training distribution harder than the test distribution.

3. **Regularization tuning** — weight decay (L2 penalty) is often underutilized. For YOLO, I bump `weight_decay` from the default 0.0005 to 0.001 if I see overfitting. Dropout is less common in modern detection backbones but label smoothing (0.1-0.2) helps calibrate overconfident predictions.

4. **Architecture downsizing** — if you're using YOLOv8m or YOLOv8l and overfitting on a small dataset, drop to YOLOv8s or even YOLOv8n. Parameter count is a strong prior for generalization when data is limited.

5. **Data quality audit** — before anything else, I check for data leakage. Overfitting that starts suspiciously early (before epoch 10) often means train/val overlap or near-duplicates split across sets.

## Q3: A safety system reports 95% accuracy on test data, but misses actual glove violations in production. Why might accuracy be misleading here, and what metric should you prioritize?

Accuracy is a dangerous metric in any imbalanced safety-critical setting. If 95% of your production frames show compliant workers, a trivial "always predict gloved" classifier achieves 95% accuracy while having 0% recall on violations — it's literally useless but looks excellent on paper.

The core issue is asymmetric cost. A false negative (missing a bare hand) could mean an injury, OSHA fine, or liability claim. A false positive (flagging a gloved hand) just means a human reviewer spends a few seconds confirming compliance. The cost ratio is easily 100:1 or higher.

I would prioritize **recall for the bare_hand class** above all else. Specifically, I track:

- **Per-class recall at the operating threshold** — not aggregate mAP, but "what fraction of actual bare hands do we catch?"
- **Precision-recall curve** to select the threshold that hits our target recall (e.g., 99% recall on bare_hand) and then accept whatever precision we get
- **False Negative Rate (FNR)** for bare_hand, monitored as a production KPI
- **mAP@0.5** as a secondary model health metric during development, not for production gating

In practice, I'd set the confidence threshold for bare_hand much lower than for gloved_hand — maybe 0.15 vs 0.4 — and let the human-in-the-loop filter handle the extra false alarms. The system is there to catch violations, not to be perfectly balanced.

## Q4: You are annotating a dataset where some hands are partially occluded by tools or overlap with other workers. How do you handle ambiguous cases consistently?

Occlusion and overlap are where annotation quality either makes or breaks a detector. Inconsistent labels create noisy gradients, and the model ends up learning the annotator's hesitation rather than the object itself.

For **occlusion**, I enforce a strict visibility rule: if ≥50% of the hand is visible, annotate the full estimated bounding box including the occluded region. This teaches the model to complete partial shapes, which is critical because occluded hands are common in real scenes. If <50% is visible, skip the annotation entirely. The exception is if your format supports an "occluded" flag (like COCO's `iscrowd` or custom attributes), in which case you can keep the box but mark it so the loss function can down-weight it.

For **overlap**, the rule is: one hand, one box. Even when two workers' hands overlap or a hand overlaps with a tool, each hand gets its own bounding box. No merging into combined regions — that destroys the instance separation the model needs to learn. Draw boxes tightly around visible portions without including large swaths of the occluding object.

I also set a clear anatomical boundary rule: wrist visible counts as a hand, fingertips-only does not. This sounds pedantic until you realize three annotators will make three different calls on a fingertip if you don't write it down.

The operational piece is just as important as the rules. I always start with a pilot annotation round of 50-100 images, calculate inter-annotator agreement (IoU overlap between annotators' boxes), and iterate on the guidelines before scaling. Spot-checking 10% of batches during production annotation catches drift early. Bad labels are exponentially more expensive to fix than good labels are to create.
