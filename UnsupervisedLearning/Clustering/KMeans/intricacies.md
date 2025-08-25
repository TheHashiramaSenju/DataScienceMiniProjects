# Guide to Imputation Estimators & Model Parameters

This document provides a detailed look at various estimators you can use for iterative imputation and a deep dive into the parameters of the powerful `HistGradientBoostingRegressor`.

--- 

## 1. Estimators for `IterativeImputer`
 
When using `IterativeImputer` from scikit-learn, the `estimator` is the machine learning model that predicts missing values based on the other features. While `RandomForestRegressor` is a strong default, other models offer different trade-offs in speed, complexity, and accuracy.

### General Purpose Estimators

This table lists the top alternative estimators you can plug into `IterativeImputer`.

| Estimator | Why Use It (Pros) | Why Not Use It (Cons) |
| :--- | :--- | :--- |
| **`BayesianRidge`** | Very fast and computationally efficient. Good default for quick iterations. | Assumes linear relationships between features, which may not hold true. |
| **`LinearRegression`** | The simplest and fastest estimator. Highly interpretable. | Can't model non-linear relationships. Sensitive to outliers. |
| **`KNeighborsRegressor`** | A non-parametric method that can capture complex, local relationships. | Computationally expensive on large datasets. Performance degrades in high dimensions. |
| **`ExtraTreesRegressor`** | Similar to Random Forest but often faster as it uses random splits for nodes. | Can sometimes be less accurate than a well-tuned Random Forest. Still a "black box" model. |
| **`DecisionTreeRegressor`** | Simple to understand and interpret. Captures non-linear patterns. | Prone to overfitting on its own. A single tree is less robust than an ensemble. |
| **`GradientBoostingRegressor`** | Often provides higher accuracy than Random Forest by correcting errors sequentially. | Can be slow to train as it builds trees one by one. More hyperparameters to tune. |
| **`HistGradientBoostingRegressor`** | A much faster version of `GradientBoostingRegressor`, optimized for large datasets. | Might be slightly less accurate than the standard `GradientBoostingRegressor`. |
| **`LGBMRegressor` (LightGBM)** | Extremely fast, memory-efficient, and often the highest-performing boosting model. | Can overfit on small datasets. Requires `pip install lightgbm`. |
| **`XGBRegressor` (XGBoost)** | The classic high-performance boosting library, known for winning competitions. | Generally slower than LightGBM. Requires `pip install xgboost`. |
| **`Lasso`** | A linear model that performs feature selection by shrinking some coefficients to zero. | Assumes linear relationships. May eliminate useful predictors if they are correlated. |

### Specialized Estimators: For Speed and Simplicity

If your primary goal is rapid prototyping or you're working with very large datasets, computationally efficient estimators are your best choice.

| Estimator | Best Use Cases | How to Use It in Your Code |
| :--- | :--- | :--- |
| **`LinearRegression`** | **Prototyping & Baseline:** When you need a quick imputation to get a baseline model running. <br> **Linear Data:** When you believe the features are linearly related. | Just import it and pass it to the imputer. <br> ```python from sklearn.linear_model import LinearRegression estimator = LinearRegression(n_jobs=-1) ``` |
| **`BayesianRidge`** | **Robust Default:** A great, fast alternative to `RandomForestRegressor`. It's more robust than standard `LinearRegression` due to regularization. <br> **Large Datasets:** Its computational efficiency is ideal for datasets where Random Forest would be too slow. | Import it from `linear_model` and swap it in. <br> ```python from sklearn.linear_model import BayesianRidge estimator = BayesianRidge() ``` |

---

## 2. `HistGradientBoostingRegressor` Parameters Explained

The `HistGradientBoostingRegressor` is a modern, high-speed algorithm. Its performance comes from binning continuous variables and its ability to handle categorical features natively. Below are its most important parameters.

| Parameter | What it Does | Common Setting / Tip |
| :--- | :--- | :--- |
| **`loss`** | The loss function to be minimized. | `'squared_error'` is standard. Use `'absolute_error'` if data has many outliers. |
| **`learning_rate`** | Controls the contribution of each tree to the ensemble. | A smaller rate (e.g., `0.05`) needs more `max_iter` but often improves accuracy. |
| **`max_iter`** | The maximum number of trees (estimators) to build. | Set high (e.g., `1000`) and let `early_stopping` find the optimal number. |
| **`early_stopping`** | Stops training when a validation score stops improving. | Keep the default `'auto'`. It prevents overfitting and saves time. |
| **`max_leaf_nodes`** | The maximum number of terminal nodes for each tree. Controls complexity. | A key parameter to tune. Values between `15` and `61` are a good starting point. |
| **`max_depth`** | The maximum depth of each individual tree. | An alternative to `max_leaf_nodes` for controlling complexity. Usually, one or the other is used. |
| **`min_samples_leaf`**| The minimum number of samples required in a leaf node. | A key regularization parameter. Increasing it (e.g., to `40` or `50`) makes the model more conservative. |
| **`l2_regularization`**| The L2 regularization penalty on leaf weights. | A small value (e.g., `0.1` to `1.0`) can help prevent overfitting on noisy data. |
| **`max_bins`** | The maximum number of bins to bucket continuous features into. | The default of `255` is almost always fine. This is the source of the model's speed. |
| **`categorical_features`** | Specifies which features are categorical, avoiding one-hot encoding. | Highly recommended. Pass a boolean mask or a list of column indices. |