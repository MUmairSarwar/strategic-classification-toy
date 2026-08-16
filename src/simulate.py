from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def make_data(n: int = 2_500, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate a reproducible two-feature classification problem."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    probability = 1 / (1 + np.exp(-(1.2 * x1 + 0.8 * x2 - 0.2)))
    target = (rng.random(n) < probability).astype(int)
    return np.column_stack([x1, x2]), target


def strategic_response(
    features: np.ndarray,
    model: LogisticRegression,
    cost_per_unit: float = 0.4,
    benefit: float = 1.0,
    max_delta: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an individual minimum-cost response to the model's decision rule.

    A currently rejected individual changes only x1 and only when the smallest
    change required for acceptance is affordable and below ``max_delta``.
    """
    changed = features.copy()
    deltas = np.zeros(len(features))
    weight = float(model.coef_[0, 0])
    if weight <= 0:
        return changed, deltas

    scores = model.decision_function(features)
    required = np.maximum(0.0, (-scores + 1e-6) / weight)
    worthwhile = (scores < 0) & (required <= max_delta) & (cost_per_unit * required < benefit)
    deltas[worthwhile] = required[worthwhile]
    changed[:, 0] += deltas
    return changed, deltas


def evaluate(model: LogisticRegression, features: np.ndarray, target: np.ndarray) -> dict[str, float]:
    probabilities = model.predict_proba(features)[:, 1]
    predictions = model.predict(features)
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "positive_decision_rate": float(predictions.mean()),
    }


def run_experiment(cost_per_unit: float = 0.4) -> tuple[dict, LogisticRegression, np.ndarray, np.ndarray, np.ndarray]:
    features, target = make_data()
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=42, stratify=target
    )
    model = LogisticRegression(max_iter=2_000, random_state=42).fit(x_train, y_train)
    baseline = evaluate(model, x_test, y_test)
    strategic_features, deltas = strategic_response(x_test, model, cost_per_unit=cost_per_unit)
    strategic = evaluate(model, strategic_features, y_test)
    metrics = {
        "sample_size": len(features),
        "test_size": len(x_test),
        "cost_per_unit": cost_per_unit,
        "benefit": 1.0,
        "max_delta": 2.0,
        "baseline": baseline,
        "strategic": strategic,
        "manipulation_rate": float((deltas > 0).mean()),
        "mean_delta_among_manipulators": float(deltas[deltas > 0].mean()),
    }
    return metrics, model, x_test, y_test, strategic_features


def _plot(metrics, model, x_test, y_test, strategic_features) -> None:
    costs = np.linspace(0.1, 1.2, 12)
    manipulation_rates, accuracies = [], []
    for cost in costs:
        changed, deltas = strategic_response(x_test, model, cost_per_unit=float(cost))
        manipulation_rates.append((deltas > 0).mean())
        accuracies.append(evaluate(model, changed, y_test)["accuracy"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].scatter(x_test[:, 0], x_test[:, 1], c=y_test, s=12, alpha=0.55, cmap="coolwarm")
    axes[0].set(title="Before strategic response", xlabel="x1 (manipulable)", ylabel="x2 (fixed)")
    axes[1].scatter(strategic_features[:, 0], strategic_features[:, 1], c=y_test, s=12, alpha=0.55, cmap="coolwarm")
    axes[1].set(title="After individual best responses", xlabel="x1 (manipulable)", ylabel="x2 (fixed)")
    axes[2].plot(costs, np.array(manipulation_rates) * 100, marker="o", label="Manipulation rate (%)")
    axes[2].plot(costs, np.array(accuracies) * 100, marker="s", label="Accuracy (%)")
    axes[2].set(title="Sensitivity to manipulation cost", xlabel="Cost per unit", ylabel="Percent")
    axes[2].legend()
    fig.suptitle("Strategic Classification: Model Decisions Change Behaviour", fontsize=17)
    fig.tight_layout()
    fig.savefig(FIGURES / "strategic_classification.svg", bbox_inches="tight")
    fig.savefig(FIGURES / "strategic_classification.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    metrics, model, x_test, y_test, strategic_features = run_experiment()
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _plot(metrics, model, x_test, y_test, strategic_features)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
