import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def make_data(n=2000):
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)

    logits = 1.2 * x1 + 0.8 * x2 - 0.2
    p = 1 / (1 + np.exp(-logits))
    y = (np.random.rand(n) < p).astype(int)

    X = np.column_stack([x1, x2])
    return X, y

def strategic_response(X, w, cost=0.4, max_delta=2.0):
    X_new = X.copy()
    w1 = w[0]

    if w1 <= 0:
        return X_new

    if (w1 - cost) <= 0:
        return X_new

    X_new[:, 0] = X_new[:, 0] + max_delta
    return X_new

def train_and_evaluate(X, y):
    n = len(y)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)

    w = clf.coef_.ravel()
    b = clf.intercept_[0]
    return clf, (X_test, y_test, proba, pred, acc, auc, w, b)

def plot_boundary(clf, X, y, title):
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(
        np.linspace(x1_min, x1_max, 200),
        np.linspace(x2_min, x2_max, 200)
    )
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    zz = clf.predict_proba(grid)[:, 1].reshape(xx1.shape)

    plt.figure()
    plt.contourf(xx1, xx2, zz, levels=20)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=10)
    plt.title(title)
    plt.xlabel("x1 (manipulable)")
    plt.ylabel("x2 (non-manipulable)")
    plt.tight_layout()

def main():
    X, y = make_data(n=2500)

    clf, (X_test, y_test, proba, pred, acc, auc, w, b) = train_and_evaluate(X, y)
    print("=== Standard setting (no strategic behavior) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC : {auc:.4f}")
    print(f"Learned weights: w={w}, b={b:.4f}")

    plot_boundary(clf, X_test, y_test, "No strategic behavior (test set)")

    X_test_strat = strategic_response(X_test, w, cost=0.4, max_delta=2.0)
    proba_s = clf.predict_proba(X_test_strat)[:, 1]
    pred_s = (proba_s >= 0.5).astype(int)

    acc_s = accuracy_score(y_test, pred_s)
    auc_s = roc_auc_score(y_test, proba_s)

    print("\n=== Under strategic behavior (users modify x1) ===")
    print(f"Accuracy: {acc_s:.4f}")
    print(f"ROC-AUC : {auc_s:.4f}")

    plot_boundary(clf, X_test_strat, y_test, "Strategic behavior: shifted x1 (test set)")

    plt.show()

if __name__ == "__main__":
    main()
