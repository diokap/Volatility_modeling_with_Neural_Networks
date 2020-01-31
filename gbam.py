import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd

def gbam(f): #f is a list, e.g. (df['Google'], 0)
    best_aic = np.inf 
    best_order = None
    
    pq_rng = range(11)
    for a in pq_rng:
        p = True
        try:
            tmp_mdl = ARIMA(f[0], order=(a,1,f[1])).fit()
            tmp_aic = tmp_mdl.aic
            for i in tmp_mdl.pvalues[1:]: #Checking the p-values of the estimated coefficients
                if i > 0.1 or np.isnan(i) == True:
                    p = False
            if p == True and tmp_aic < best_aic:
                best_aic = round(tmp_aic, 4)
                best_order = (a, f[1])
        except: continue
    
    return best_aic, best_order