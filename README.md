# Strategic Classification (Toy Model)

This project demonstrates **strategic behavior in machine learning**:
users can modify a manipulable feature to improve their predicted outcome, which can
change model performance and decision boundaries.

## Idea
- Train a classifier on 2D synthetic data.
- Assume feature **x1** is manipulable.
- Users strategically increase x1 if the benefit outweighs a cost.
- Evaluate the same trained model under strategic behavior.

## Run (local)
```bash
pip install -r requirements.txt
python src/simulate.py
