# Strategic Classification and Behavioural Response

A small, reproducible experiment showing why predictive models cannot always treat people as passive data points. When a scoring rule affects access to an outcome, people may change features that the model rewards.

![Experiment overview](reports/figures/strategic_classification.svg)

## Question

How does an unchanged logistic-regression model perform after rejected individuals can modify one feature to cross its decision boundary?

The experiment uses two synthetic features so the mechanism remains visible:

- `x1` can be changed at a personal cost.
- `x2` cannot be changed.
- A rejected individual makes only the minimum affordable change needed for acceptance.
- The true label does not change after manipulation.

This is a controlled mathematical demonstration, not a claim about a specific real-world population.

## What is measured

- Accuracy and ROC AUC before and after strategic response
- Positive-decision rate
- Share of individuals who manipulate
- Mean change among manipulators
- Sensitivity of manipulation and accuracy to the assumed cost

Measured outputs are committed in [`reports/metrics.json`](reports/metrics.json). Run the script to reproduce them:

At the default cost, **50.2%** of test individuals changed `x1`. The positive-decision rate increased from **46.1% to 96.3%**, while accuracy fell from **70.4% to 50.6%** and ROC AUC from **0.800 to 0.773**. These are simulated results under the stated assumptions.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python -m src.simulate
```

## Why it matters

Credit, admissions, hiring and platform-ranking models can change the behaviour they measure. A model that performs well on historical data may behave differently after its decision rule becomes actionable. Useful follow-up work would retrain against anticipated responses, add heterogeneous costs and evaluate fairness across groups.

## Author

Muhammad Umair Sarwar — Mathematics in Data Science, TU Darmstadt
