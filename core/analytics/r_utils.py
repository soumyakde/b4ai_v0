import os
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.conversion import localconverter


# Optional: set R environment variable explicitly
os.environ["R_HOME"] = r'"C:/Program Files/R/R-4.5.2"'  # adjust to your R install

def to_r(df):
    """Convert pandas DataFrame to R DataFrame safely for Windows/Streamlit."""
    with localconverter(pandas2ri.converter):
        return pandas2ri.py2rpy(df)

def to_pd(r_df):
    """Convert R DataFrame back to pandas."""
    with localconverter(pandas2ri.converter):
        return pandas2ri.rpy2py(r_df)