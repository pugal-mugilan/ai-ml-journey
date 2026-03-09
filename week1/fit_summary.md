# How scikit-learn's `fit()` Works — Plain English Explanation

The `fit()` method is used to make the model understand the pattern between existing features and targets. We pass `X`, which contains features like location and number of bedrooms, and `y`, which contains the target values like house prices. `fit()` analyzes the relationship between them and stores the results.

It stores `coef_`, which captures the effect of each feature on the house price. `rank_` tells us how many columns carry truly independent information — for example, if two columns represent the same data in different units, `rank_` would count them as one. `singular_` stores how much useful information each feature contains. `predict()` then uses these stored values to make predictions on new, unseen data.

These attributes end with an underscore because scikit-learn follows a convention: the trailing `_` signals that these are created by `fit()` and can only be accessed after calling it.

There's also a reason `fit()` and `predict()` are separate. `fit()` finds the pattern, and `predict()` applies it. If they were combined, the model would need to relearn the pattern for every new prediction. Instead, we learn the pattern once and reuse it as many times as needed — a much more efficient design.