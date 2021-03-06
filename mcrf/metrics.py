from collections import defaultdict
from functools import reduce


def prf1_for_tagging(gold_ents, pred_ents, eps=1e-6):
    measure_results = {
        "micro": defaultdict(lambda: {"p": 0.0, "r": 0.0, "f1": 0.0}),
        "macro": defaultdict(lambda: {"p": 0.0, "r": 0.0, "f1": 0.0}),
    }
    global_tp = global_fp = global_fn = 0
    for gold_ents, pred_ents in zip(gold_ents, pred_ents):
        gold_ents = set(reduce(lambda x, y: x + y, gold_ents.values(), []))
        pred_ents = set(reduce(lambda x, y: x + y, pred_ents.values(), []))
        intersection = gold_ents & pred_ents
        global_tp += len(intersection)
        global_fp += len(pred_ents - intersection)
        global_fn += len(gold_ents - intersection)

    p = measure_results['micro']['overall']["p"] = global_tp / (global_tp + global_fp + eps)
    r = measure_results['micro']['overall']["r"] = global_tp / (global_tp + global_fn + eps)
    measure_results['micro']['overall']["f1"] = 2 * p * r / (p + r + eps)
    return measure_results
