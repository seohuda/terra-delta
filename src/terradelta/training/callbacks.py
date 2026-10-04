"""Validation observation and optional early stopping; no training side effects."""
import math


def metric_value(metrics, key):
    if key in metrics:
        value = metrics[key]
    else:
        value = metrics
        for part in key.split("."):
            value = value[part]
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite validation metric {key}")
    return result


def observe_metric(metrics, config, best_metric=None, bad_validations=0):
    key = config.get("metric")
    if not key:
        return best_metric, bad_validations, False, False
    value = metric_value(metrics, key)
    delta = float(config.get("min_delta", 0))
    improved = best_metric is None or (value > best_metric + delta if config.get("mode", "max") == "max"
                                      else value < best_metric - delta)
    if improved:
        best_metric, bad_validations = value, 0
    else:
        bad_validations += 1
    stop = bool(config.get("enabled", False)) and bad_validations >= config.get("patience", 5)
    return best_metric, bad_validations, improved, stop
