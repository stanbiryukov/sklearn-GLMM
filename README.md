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


Under the hood, the class relies on the R libraries listed below, so for it to work properly they must be installed in the R environment linked to rpy2. R's pacman will attempt to install any missing libraries but rstan usually requires system specific configuration.
```R
library(pacman)
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
ml = skGLMM(x_scaler = StandardScaler(), y_scaler = FunctionTransformer(validate=True), 
            r_call = "brm(remission ~ IL6 + CRP + CancerStage + LengthofStay + Experience + (1 | DID), data = df, family = bernoulli(), algorithm='sampling', iter = 1000, chains = 4, cores = 4)")

ml.fit(df[['IL6', 'CRP', 'CancerStage', 'LengthofStay', 'Experience', 'DID']], df['remission'])
phat = ml.predict(df[['IL6', 'CRP', 'CancerStage', 'LengthofStay', 'Experience', 'DID']])
# Support for parallel prediction and sampling from posterior for BRMS and LME4 models
phatdraws = ml.predict(df[['IL6', 'CRP', 'CancerStage', 'LengthofStay', 'Experience', 'DID']], n_draws=1000, parallel=True)

# Also returns R style summary
ml.summary()
```
```
 Family: bernoulli 
  Links: mu = logit 
Formula: remission ~ IL6 + CRP + CancerStage + LengthofStay + Experience + (1 | DID) 
   Data: df (Number of observations: 8525) 
Samples: 4 chains, each with iter = 1000; warmup = 500; thin = 1;
         total post-warmup samples = 2000

Group-Level Effects: 
~DID (Number of levels: 407) 
              Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
sd(Intercept)     1.99      0.10     1.80     2.20 1.00      401      746

Population-Level Effects: 
             Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
Intercept       -1.48      0.10    -1.68    -1.28 1.00      259      615
IL6             -0.15      0.03    -0.21    -0.09 1.00     2717     1470
CRP             -0.06      0.03    -0.12    -0.00 1.00     1867     1228
CancerStage     -0.29      0.03    -0.35    -0.22 1.00     1935     1258
LengthofStay    -0.31      0.03    -0.38    -0.25 1.00     1905     1377
Experience       0.49      0.12     0.25     0.73 1.02      130      136

Samples were drawn using sampling(NUTS). For each parameter, Eff.Sample 
is a crude measure of effective sample size, and Rhat is the potential 
scale reduction factor on split chains (at convergence, Rhat = 1).
```
