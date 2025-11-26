# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 10:32:34 2022

@author: Diogo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc
from scipy.io import loadmat
from scipy import stats
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.metrics import r2_score
from sysidentpy.metrics import mean_absolute_error
from sysidentpy.metrics import explained_variance_score
from sysidentpy.metrics import forecast_error
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.residues.residues_correlation import compute_residues_autocorrelation, compute_cross_correlation
from sysidentpy.parameter_estimation import LeastSquares

def mcor(y,yhat):
    Y1 = y-y.mean();
    Y12 = Y1**2;
    Y2 = yhat-y.mean();
    Y22 = Y2**2;
    r2 = Y22.sum()/Y12.sum()
    return r2
def mse(y,yhat):
    MSE = np.square(y-yhat).mean()
    return MSE

def unorm(u):
    umax = np.max(u)
    umin = np.min(u)
    fat = np.abs(umax-umin)
    un = u/fat
    return un

def ynorm(y):
    ymax = np.max(y)
    ymin = np.min(y)
    fat = np.abs(ymax-ymin)
    yn = (y-y[0])/fat
    return yn

DATA_train = loadmat('pythondatatrain.mat')
x_train = DATA_train['u']
y_train = DATA_train['y']/10
DATA_val = loadmat('pythondataval.mat')
x_valid = DATA_val['u']
y_valid = DATA_val['y']/10


##Normalização dos dados



basis_function = Polynomial(degree=3)
estimator = LeastSquares()
model = FROLS(
    order_selection=True,
    n_info_values=10,
    ylag=3, xlag=3,
    info_criteria='aic',
    estimator=estimator,
    basis_function=basis_function,
    err_tol=None,
)
model.fit(X=x_train, y=y_train)
r = pd.DataFrame(
    results(
        model.final_model, model.theta, model.err,
        model.n_terms, err_precision=8, dtype='sci'
        ),
    columns=['Regressors', 'Parameters', 'ERR'])
print(r)


##Resultados da simulação com dados de treinamento
yhat_train = model.predict(X=x_train, y=y_train)
rrse = root_relative_squared_error(y_train, yhat_train)
print('RRSE de treinamento=',rrse)
R2_train = r2_score(y_train,yhat_train)
print('R2 de treinamento=',R2_train)
MSE = mean_absolute_error(y_train, yhat_train)
print('Mse de treinamento',MSE)
res = forecast_error(y_train, yhat_train)
fig, ax = plt.subplots()
ax.plot(res)
ax.set(title='Residue')
ax.grid()
plt.show()     
EVS = explained_variance_score(y_train, yhat_train)

plot_results(y=y_train, yhat=yhat_train,n=33900,marker="",model_marker="")
ee = compute_residues_autocorrelation(y_train, yhat_train)
plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
x1e = compute_cross_correlation(y_train, yhat_train, x_train)
plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")

##Resultados da simulação com dados de treinamento
yhat_valid = model.predict(X=x_valid, y=y_valid)
rrse = root_relative_squared_error(y_valid, yhat_valid)
print('RRSE de validação=',rrse)
R2_valid = r2_score(y_valid,yhat_valid)
print('R2 de validação=',R2_valid)
MSE = mse(y_valid, yhat_valid)
print('Mse de validação=',MSE)
res = y_valid-yhat_valid
fig, ax = plt.subplots()
ax.plot(res)
ax.set(title='Residue')
ax.grid()

plt.show()     



plot_results(y=y_valid, yhat=yhat_valid,n=33900,marker="",model_marker="")
ee = compute_residues_autocorrelation(y_valid, yhat_valid)
plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
x1e = compute_cross_correlation(y_valid, yhat_valid, x_valid)
plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")
plt.figure()
minY = min(min(y_valid),min(yhat_valid))
maxY = max(max(y_valid),max(yhat_valid))
plt.scatter(y_valid,yhat_valid,c='red',label='Prediction')
plt.plot([minY, maxY], [minY, maxY], color = 'black', linewidth = 2,label='Perfect model')
plt.xlabel('Real')
plt.ylabel('Prediction')
plt.grid()
plt.legend()
plt.show()  
results = {'x_valid':x_valid,'x_train':x_train,'y_hat_train':yhat_train,'y_hat_valid':yhat_valid}
sc.io.savemat('resultados_narmax.mat',results)