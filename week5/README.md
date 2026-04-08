# Classification Models

**Goal:** Move from predicting numbers (regression) to predicting categories (classification). Three models, two paradigms, one pipeline.

---

## What I Built

###  Logistic Regression from Scratch
Built logistic regression on tumor data (benign vs malignant). Only **3 things changed** from linear regression:
1. Added sigmoid to squish output to 0-1
2. Replaced MSE with binary cross-entropy (MSE has a ceiling with sigmoid)
3. Derivative constant changed from `2/n` to `1/n`

Everything else — the training loop, `X.T @ error`, the update rule — identical to linear regression.

###  Classification Metrics
Built a complete metrics toolkit: confusion matrix, accuracy, precision, recall, F1 score. Key insight: **accuracy lies with imbalanced data** (95% accuracy on 95/5 split = predicting majority class every time). Business context decides whether to optimize precision (spam filter) or recall (cancer detection). The threshold is the knob that trades one for the other.

###  Decision Trees + Random Forests
A completely different paradigm — **no gradient descent, no weights, no derivatives**. Trees learn by asking yes/no questions using Gini impurity to measure "how mixed is this group?" Random forests build 100 trees on random data subsets and take majority vote — noise gets outvoted.

| Model | Train Accuracy | Test Accuracy |
|---|---|---|
| Overfit Tree | 1.000 | 0.960 |
| Pruned Tree (depth=2) | 0.980 | **0.970** |
| Random Forest | 1.000 | 0.960 |

### 3-Model Comparison Pipeline
Full pipeline on stock prediction data (5 features, 500 samples). Same dataset, three models, honest evaluation.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.68 | 0.75 | 0.77 | 0.76 |
| Decision Tree (d=3) | 0.73 | 0.76 | 0.85 | 0.80 |
| Random Forest (d=3) | 0.73 | 0.76 | 0.85 | 0.80 |

**Key trade-off:** Forest predicts better but explains worse. LR predicts worse but explains better. Choose based on whether you need to explain or just predict.

###  sklearn Comparison
Rebuilt all three models using sklearn. Results matched scratch implementations, confirming the from-scratch code was correct. sklearn's logistic regression got a small accuracy bump (0.70 vs 0.68) from its default Ridge penalty (`C=1.0`).

---

## Key Concepts Learned

**Two ML Paradigms:**
| | Optimization-based | Tree-based |
|---|---|---|
| How it learns | Gradient descent | Recursive Gini splits |
| Models | Linear Reg → Logistic Reg → Neural Nets | Decision Tree → Random Forest → XGBoost |
| Overfitting fix | Regularization (Ridge/Lasso) | Pruning + Ensembles |

**Evaluation Framework:**
- Never trust accuracy alone — check precision, recall, F1
- Always compare against a baseline (majority class predictor)
- Always evaluate on unseen test data (train/test split)
- Business context decides which metric matters most

---

## Files

| File                                 | Description |
|--------------------------------------|---|
| `1_logistic_regression_scratch.py`   | Logistic regression from scratch — sigmoid, cross-entropy, gradient descent |
| `2_classification_metrics.py`        | Confusion matrix, precision, recall, F1, threshold sweep, accuracy paradox |
| `3_decision_trees_random_forests.py` | Gini impurity, overfitting demo, pruning, random forest voting |
| `4_three_model_pipeline.py`          | Full 3-model comparison on stock data with normalization + feature importance |
| `sklearn_comparison.py`              | sklearn versions confirming scratch results |

---

 5:** Statistical significance → now we evaluate with precision/recall instead of p-values, but the principle is the same: don't trust surface-level numbers.