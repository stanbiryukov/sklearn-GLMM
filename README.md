# sklearn-GLMM
scikit-learn wrapper for generalized linear mixed model methods in R

This is a lightweight wrapper that enables fitting generalized linear multivariate multilevel models from python via R.
It easily enables fitting bayesian models via brms calls: https://github.com/paul-buerkner/brms

It is flexible enough to extend to other R based models and is designed to be as compatible with scikit-learn syntax as possible.

It's specifically built against rpy2==2.8.6 to enable both python2 and 3 support.

Installation:
```python
pip install git+https://github.com/stanbiryukov/sklearn-GLMM
```


Under the hood, the class relies on the following R libraries, so for it to work properly they must be installed in the R environment linked to rpy2:
```R
library(rstan)
library(parallel)
library(brms)
library(lme4)
library(feather)
library(data.table)
library(dplyr)
library(merTools)
library(pbmcapply)
```

Demonstrate by example:
```python
# df is your pandas dataframe with predictors and target column
import pandas as pd
df = pd.read_csv('https://stats.idre.ucla.edu/stat/data/hdp.csv')
df = df.apply(lambda x: pd.factorize(x)[0] if np.issubdtype(x.dtype, np.number) is False else x) # factorize some columns

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
