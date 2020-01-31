import numpy as np
from arch import arch_model
import pandas as pd

def gbgm(f): #f is a list, e.g. (arima.resid, 0, 'GARCH')
    best_aic = np.inf 
    best_order = None
    
    pq_rng = range(11)
    for a in pq_rng:
        p = True
        try:
            tmp_mdl = arch_model(f[0], p=a, q=f[1], vol=f[2], mean='Zero', rescale=True).fit()
            tmp_aic = tmp_mdl.aic
            for i in tmp_mdl.pvalues: #Checking the p-values of the estimated coefficients
                if i > 0.1 or np.isnan(i) == True:
                    p = False
            if p == True and tmp_aic < best_aic:
                best_aic = round(tmp_aic, 4)
                best_order = (a, f[1])
        except: continue
    
    return best_aic, best_order