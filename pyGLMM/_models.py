import numpy as np
import pandas as pd
import rpy2.robjects.packages as r
import tempfile
import time

from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import STAP
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import FunctionTransformer, StandardScaler

class skGLMM(BaseEstimator, RegressorMixin):
    """
    Wrapper for BRMS & LME4 type models. 
    Designed to be flexible to work with other base stats regressors (glm) and more R models.
    Built against rpy2==2.8.6 for python2 support.
    Since this is based on a formula in r_call, the caveat is that dataframe columns need to be passed to the methods, not numpy arrays.
    """

    def __init__(
        self, r_call, x_scalar=StandardScaler(), y_scalar=FunctionTransformer(validate=True)
    ):
        self.r_call = r_call
        self.x_scalar = x_scalar
        self.y_scalar = y_scalar

    def get_r(self):
        r_fct_string = """
        fit_ <- function(r_call, dfpath) {
        library(rstan)
        library(parallel)
        library(brms)
        library(lme4)
        library(feather)
        library(data.table)
        rstan_options(auto_write = TRUE)
        options(mc.cores = parallel::detectCores())
        # print(R.Version())
        df = read_feather(dfpath)
        df = data.table(df)
        # df = df[, names(df) := lapply(.SD, as.numeric)]
        fit = eval(parse(text=r_call))
        print(summary(fit))
        fpath = tempfile(pattern = "", fileext = ".rds")
        # save the model
        saveRDS(fit, fpath)
        # save summary
        capture.output( summary(fit), file=gsub(".rds", ".txt", fpath), append=TRUE)
        return (fpath)
        }
        predict_ <- function(model, newdfpath, n_draws, parallel) {
        library(data.table)
        library(parallel)
        library(dplyr)
        library(feather)
        library(merTools)
        library(rstan)
        library(pbmcapply)
        numCores <- detectCores() - 1
        parpred <- function(dfc, model) {
            return (data.table(predict(model, newdata=dfc, type='response')))[,1]
        }
        parpreddraws <- function(dfc, model, n_draws) {
            return (data.table(t(predict(model, newdata=dfc, nsamples = n_draws, summary = FALSE))))
        }
        rstan_options(auto_write = TRUE)
        options(mc.cores = parallel::detectCores())
        mdl = readRDS(model)
        newdf = read_feather(newdfpath)
        newdf = data.table(newdf)
        spl.dt <- split( newdf , cut(1:nrow(newdf), min(nrow(newdf), numCores)) )
        if (parallel==TRUE) {
            if (n_draws > 0) {
                if (grepl('brms', class(mdl), fixed=TRUE)) {
                    # preddraws = data.table(t(predict(mdl, newdata=newdf, nsamples = n_draws, summary = FALSE)))
                    preddraws = rbindlist(pbmclapply(spl.dt, parpreddraws, model=mdl, n_draws = n_draws, mc.cores = numCores))
                    preddraws = preddraws %>% dplyr::rename_at(vars(starts_with('V')),  funs(sub('V', 'draw_', .)))
                } else {
                    pred = predictInterval(mdl, newdata = newdf, n.sim=n_draws, include.resid.var = TRUE, .parallel=TRUE, returnSims=TRUE, level = .95, seed = 842)
                    preddraws = data.table(attr(pred, "sim.results"))
                    preddraws = preddraws %>% dplyr::rename_at(vars(starts_with('V')),  funs(sub('V', 'draw_', .)))
                }
            } else {
                preddraws = rbindlist(pbmclapply(spl.dt, parpred, model=mdl, mc.cores = numCores))[,1]
            }
        } else {
            if (n_draws > 0) {
                if (grepl('brms', class(mdl), fixed=TRUE)) {
                    preddraws = data.table(t(predict(mdl, newdata=newdf, nsamples = n_draws, summary = FALSE)))
                    preddraws = preddraws %>% dplyr::rename_at(vars(starts_with('V')),  funs(sub('V', 'draw_', .)))
                } else {
                    pred = predictInterval(mdl, newdata = newdf, n.sim=n_draws, include.resid.var = TRUE, .parallel=FALSE, returnSims=TRUE, level = .95, seed = 842)
                    preddraws = data.table(attr(pred, "sim.results"))
                    preddraws = preddraws %>% dplyr::rename_at(vars(starts_with('V')),  funs(sub('V', 'draw_', .)))
                }
            } else {
                preddraws = (data.table(predict(mdl, newdata=newdf, type='response')))[,1]
            }
        }
        # predpath = tempfile(pattern = "", fileext = ".feather")
        # write_feather(preddraws, predpath)
        # use r2py io for this part: https://issues.apache.org/jira/browse/ARROW-1907
        return(preddraws)
        }
        """
        return r_fct_string

    def fit(self, X_in, z_in):
        self.dir = tempfile.gettempdir()
        depv = self.r_call.split("~")[0]
        self.dep_var = depv.split("(")[1].strip()
        self.y_scalar.fit(z_in.values.reshape(-1, 1))
        yvals = self.y_scalar.transform(z_in.values.reshape(-1, 1))
        self.x_scalar.fit(X_in.values)
        xvals = self.x_scalar.transform(X_in.values)
        dfin = pd.concat(
            [
                pd.DataFrame(yvals, columns=[self.dep_var]),
                pd.DataFrame(xvals, columns=X_in.columns),
            ],
            axis=1,
        )
        self.dfpath = self.dir + "/data_in.feather"
        dfin.to_feather(self.dfpath)
        # print(dfin.head())
        # r_dfin = pandas2ri.py2ri(dfin)
        self.ml = STAP(self.get_r(), "r_fct_string")
        print("Starting Fit.")
        start = time.time()
        self.ml_ = self.ml.fit_(self.r_call, dfpath=self.dfpath)
        print("R Fit Done. It took %.0f seconds" % (time.time() - start))
        f = open(self.ml_[0].replace(".rds", ".txt"))
        self._summary = f.read().splitlines()
        f.close()

    def predict(self, X, n_draws=0, parallel=False):
        X_out = self.x_scalar.transform(X)
        # r_dfout = pandas2ri.py2ri(pd.DataFrame(X_out, columns = X.columns))
        dfout = pd.DataFrame(X_out, columns=X.columns)
        self.dfoutpath = self.dir + "/data_out.feather"
        dfout.to_feather(self.dfoutpath)
        out_ = self.ml.predict_(self.ml_, self.dfoutpath, n_draws, parallel)
        pred = pandas2ri.ri2py_dataframe(out_)
        return self.y_scalar.inverse_transform(pred.values)

    def score(self, X, y):
        """
        Root Mean Squared Error via numpy
        """
        return -np.sqrt(
            np.mean(
                np.power(
                    self.predict(X, n_draws=0).reshape(-1, 1) - y.reshape(-1, 1), 2
                )
            )
        )

    def summary(self):
        for line in self._summary:
            print(line)
