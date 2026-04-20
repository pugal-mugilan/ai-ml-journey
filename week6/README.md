# Gradient Boosting + Unsupervised Learning

**Deliverable:** Complete ML pipeline showing how the three algorithm families (optimization-based, tree-based, unsupervised) combine into one end-to-end workflow.

## Pipeline architecture

```
  ┌─────────────────────────────┐
  │ 1. Load and scale data      │  500 rows × 5 features, 400/100 split
  └──────────────┬──────────────┘
                 │
                 ▼
  ┌─────────────────────────────┐
  │ 2. PCA for exploration      │  Project to 2D → check structure,
  │                             │  measure explained variance
  └──────────────┬──────────────┘
                 │
                 ▼
  ┌─────────────────────────────┐
  │ 3. Isolation Forest         │  Flag outliers (no labels needed),
  │                             │  produce a cleaned training set
  └──────────────┬──────────────┘
                 │
                 ▼
  ┌─────────────────────────────┐
  │ 4. Gradient Boosting        │  Train on: raw / PCA / cleaned
  │                             │  variants, compare against LR + RF
  └──────────────┬──────────────┘
                 │
                 ▼
  ┌─────────────────────────────┐
  │ 5. Model comparison         │  Accuracy / Precision / Recall / F1
  └─────────────────────────────┘
```

## Files

| File | Purpose |
|---|---|
| `full_pipeline.py` | Runnable end-to-end pipeline |
| `models_comparison.md` | Results table + interpretation |
| `Week_6_notes.md` | Session-by-session concept notes (Days 1–5) |

## Key findings

1. **PCA on uncorrelated features loses information.** PC1+PC2 captured only 43% of variance on this dataset. Gradient Boosting trained on the PCA-reduced features lost 10 accuracy points (0.900 → 0.810). PCA compresses *redundancy* — if features are near-independent, there's nothing to compress.

2. **Isolation Forest as a data cleaner, not a model.** Flagged 5% of training rows as anomalies. Dropping them gave Gradient Boosting +1 accuracy point and matched Logistic Regression performance. In production this is the right framing — anomaly detection upstream of your supervised model, not as a replacement for it.

3. **Match the model to the data-generating process.** Data was linearly generated → Logistic Regression won (0.910). Tree ensembles and boosting landed at 0.900. The lesson: before reaching for the fanciest algorithm, check whether the simplest one already fits.

## Models covered 

**Optimization-based:** Linear Regression → Logistic Regression 
**Tree-based:** Decision Tree → Random Forest  → Gradient Boosting 
**Unsupervised:** K-Means, DBSCAN, PCA, Isolation Forest 

Full comparison table in `models_comparison.md`.

## Run it

```bash
python full_pipeline.py
```

Dependencies: `numpy`, `scikit-learn`. Uses a fixed random seed (`42`) throughout; results are reproducible.

