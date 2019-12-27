# sklearn-GLMM
scikit-learn wrapper for generalized linear mixed model methods in R

This is a lightweight wrapper that enables fitting generalized linear multivariate multilevel models from python.
It easily enables fitting bayesian models via brms calls: https://github.com/paul-buerkner/brms
It is flexible enough to extend to other R based models and is designed to be as compatible to scikit-learn syntax as possible.

Installation
```python
pip install git+https://github.com/stanbiryukov/sklearn-GLMM
```

```python
# df is your pandas dataframe with predictors and target column
from pyGLMM import skGLMM
from sklearn.preprocessing import FunctionTransformer, StandardScaler
ml = skGLMM(x_scalar = StandardScaler(), y_scalar = FunctionTransformer(validate=True), 
  r_call = "brm(remission ~ IL6 + CRP + CancerStage + LengthofStay + Experience + (1 | DID), 
  silent = TRUE, data = df, family = bernoulli(), refresh=0, algorithm='sampling', iter = 1000, chains = 4, cores = 4)")

ml.fit(df[['IL6', 'CRP', 'CancerStage', 'LengthofStay', 'Experience', 'DID']], df['remission'])
phat = ml.predict(df[['IL6', 'CRP', 'CancerStage', 'LengthofStay', 'Experience', 'DID']])
# Support for parallel prediction and sampling from posterior for BRMS and LME4 models
phatdraws = ml.predict(df[['IL6', 'CRP', 'CancerStage', 'LengthofStay', 'Experience', 'DID']], n_draws=1000, parallel=True)

# Also returns R style summary
ml.summary()
```
