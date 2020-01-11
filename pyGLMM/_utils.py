import pandas as pd

def r_formula(df, dependent_var, re=None, excluded_cols=None, include_intercept=True):
    """
    Create a lme4 formula syntax from pandas dataframe. 
    """
    df_columns = list(df.columns.values)
    df_columns.remove(dependent_var)
    if excluded_cols is not None:
        for col in excluded_cols:
            try:
                df_columns.remove(col)
            except ValueError:
                pass
    if include_intercept in [1]:
        frm = dependent_var + " ~ " + " + ".join(df_columns)
    elif include_intercept in [0]:
        frm = dependent_var + " ~ -1 + " + " + ".join(df_columns)
    if re:
        frm = frm + " + " + re
    return frm
