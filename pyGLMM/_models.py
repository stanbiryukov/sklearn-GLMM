import numpy as np
import os
import pandas as pd
import tempfile
import time
import uuid

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

    Parameters
    ----------
    r_call : {string}, data parameter in R formula must be 'df'.
        example: 'brm(count ~ zAge + zBase * Trt + (1|patient), data = df, family = poisson())'
    outdir : {string}, directory where to save R/Python IO and R model. Defaults to tempfile's tempdir search, but note that for the 
                        model to be pickled and used later, this path must exist since the R model is saved in this location. 
    x_scaler : {sklearn.preprocessing object}.
        Transforms done to predictors before R model invoked and inverted upon predict.
        default: StandardScaler()
    y_scaler : {sklearn.preprocessing object}. 
        Transforms done to target before R model invoked and inverted upon predict.
        default: FunctionTransformer(validate=True)
    pacman_call: {string}, pacman load to extend wrapper to other model types.
        default: 'pacman::p_load(lme4)'
    """

    def __init__(
        self,
        r_call,
        outdir=tempfile.gettempdir(),
        pacman_call="pacman::p_load(lme4)",
        x_scaler=StandardScaler(),
        y_scaler=FunctionTransformer(validate=True),
    ):
        super(skGLMM, self).__init__()
        self.r_call = r_call
        self.outdir = outdir
        self.pacman_call = pacman_call
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        os.system("mkdir -p -m 777 {}".format(outdir))

    def get_r(self):
        r_fct_string = """
        fit_ <- function(r_call, dfpath, pcmstr) {
        set.seed(8502) # set random seed in fitting process
        library(pacman)
        pacman::p_load(rstan, parallel, brms, lme4, feather, data.table, dplyr, merTools, pbmcapply, lme4)
        eval(parse(text=pcmstr))
        rstan_options(auto_write = TRUE)
        options(mc.cores = parallel::detectCores())
        # print(R.Version())
        df = read_feather(dfpath)
        df = data.table(df)
        # df = df[, names(df) := lapply(.SD, as.numeric)]
        fit = eval(parse(text=r_call))
        print(summary(fit))
        fpath = paste0(dirname(dfpath), '/', tools::file_path_sans_ext(basename(dfpath)), 'model.rds')
        # save the model
        saveRDS(fit, fpath)
        # save summary
        capture.output( summary(fit), file=gsub(".rds", ".txt", fpath), append=TRUE)
        return (fpath)
        }
        predict_ <- function(model, newdfpath, n_draws, parallel, pcmstr) {
        set.seed(2063) # different random seed in predict for reproducibility
        library(pacman)
        pacman::p_load(rstan, parallel, brms, lme4, feather, data.table, dplyr, merTools, pbmcapply, lme4)
        eval(parse(text=pcmstr))
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

    def get_ml(self):
        ml = STAP(self.get_r(), "r_fct_string")
        return ml

    def fit(self, X_in, z_in):
        depv = self.r_call.split("~")[0]
        self.dep_var = depv.split("(")[1].strip()
        self.y_scaler.fit(z_in.values.reshape(-1, 1))
        yvals = self.y_scaler.transform(z_in.values.reshape(-1, 1))
        self.x_scaler.fit(X_in.values)
        xvals = self.x_scaler.transform(X_in.values)
        dfin = pd.concat(
            [
                pd.DataFrame(yvals, columns=[self.dep_var]),
                pd.DataFrame(xvals, columns=X_in.columns),
            ],
            axis=1,
        )
        dfpath = "{}/{}.feather".format(self.outdir, uuid.uuid4())
        dfin.to_feather(dfpath)
        ml = self.get_ml()
        print("Starting Fit.")
        start = time.time()
        self.ml_ = ml.fit_(self.r_call, dfpath=dfpath, pcmstr=self.pacman_call)
        print("R Fit Done. It took %.0f seconds" % (time.time() - start))
        f = open(self.ml_[0].replace(".rds", ".txt"))
        self._summary = f.read().splitlines()
        f.close()
        # clean up I/O
        os.remove(dfpath)
        os.remove(self.ml_[0].replace(".rds", ".txt"))

    def predict(self, X, n_draws=0, parallel=False):
        X_out = self.x_scaler.transform(X)
        dfout = pd.DataFrame(X_out, columns=X.columns)
        dfoutpath = "{}/{}.feather".format(self.outdir, uuid.uuid4())
        dfout.to_feather(dfoutpath)
        ml = self.get_ml()
        out_ = ml.predict_(self.ml_, dfoutpath, n_draws, parallel, self.pacman_call)
        pred = pandas2ri.ri2py_dataframe(out_)
        os.remove(dfoutpath)
        return self.y_scaler.inverse_transform(pred.values)

    def score(self, X, y):
        """
        Root Mean Squared Error via numpy
        """
        return -np.sqrt(
            np.mean(
                np.power(
                    np.subtract(
                        self.predict(X, n_draws=0).reshape(-1, 1),
                        y.values.reshape(-1, 1),
                    ),
                    2,
                )
            )
        )

    def summary(self):
        for line in self._summary:
            print(line)

    def get_params(self, deep=True):
        return {"y_scaler": self.y_scaler, "x_scaler": self.x_scaler}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
