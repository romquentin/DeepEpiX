import numpy as np


def compute_matches(model_onsets, gt_onsets, delta):
    model_onsets = list(model_onsets)
    gt_onsets = list(gt_onsets)

    true_positive = 0
    false_positive = 0
    false_negative = 0

    matched_gt = set()
    tp_distances = []
    fp_distances = []
    fn_distances = []

    for m in model_onsets:
        matched = False
        for g in gt_onsets:
            if abs(m - g) <= delta and g not in matched_gt:
                true_positive += 1
                matched_gt.add(g)
                tp_distances.append(abs(m - g))
                matched = True
                break
        if not matched:
            false_positive += 1
            # Distance to closest ground truth
            closest_gt = min(gt_onsets, key=lambda g: abs(m - g)) if gt_onsets else None
            if closest_gt is not None:
                fp_distances.append(abs(m - closest_gt))

    for g in gt_onsets:
        if g not in matched_gt:
            false_negative += 1
            # Distance to closest model prediction
            closest_model = (
                min(model_onsets, key=lambda m: abs(m - g)) if model_onsets else None
            )
            if closest_model is not None:
                fn_distances.append(abs(g - closest_model))

    return (
        true_positive,
        false_positive,
        false_negative,
        tp_distances,
        fp_distances,
        fn_distances,
    )


def get_distance_stats(distances):
    if not distances:
        return {"mean": "N/A", "std": "N/A", "min": "N/A", "max": "N/A"}
    arr = np.array(distances)
    return {
        "mean": f"{np.mean(arr):.3f}",
        "std": f"{np.std(arr):.3f}",
        "min": f"{np.min(arr):.3f}",
        "max": f"{np.max(arr):.3f}",
    }
