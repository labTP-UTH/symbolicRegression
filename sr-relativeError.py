import os
import glob, os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy, sys, getopt
import scienceplots
import matplotlib.ticker
from matplotlib.ticker import AutoMinorLocator
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error
import sympy
from sympy import*
from numpy import savetxt
from pysr import PySRRegressor


#
# main script
#

# preliminaries
SCALE_DATASET_     = true
PARTITION_DATASET_ = true


# -----------------------------------------------------------------#
# inputs

ifile=-1
for file in glob.glob("*.pkl"):
    ifile = ifile + 1
#    print(file)

if(ifile>0):
    print('Multiple pkl fles are found! Please make sure your input is correct!!!')
    
for fname in os.listdir('.'):
    if fname.endswith('.pkl'):
        #print(fname)
        fin=fname
        break
else:
    print("file pkl does not exist!!! continue with tmp.pkl if exists...")
    fin='tmp.pkl'

df1 = pd.read_csv('ShortTubeData.csv')
data0 = numpy.loadtxt( 'ShortTubeData.csv', skiprows = 1, delimiter=',', dtype=float )

# -----------------------------------------------------------------#

x0_ = np.zeros((df1.shape[0]))
x1_ = np.zeros((df1.shape[0])) 
x2_ = np.zeros((df1.shape[0]))
x3_ = np.zeros((df1.shape[0]))
Equation_ = np.zeros((x0_.shape[0]))

x0_[:] = data0[:,0]
x1_[:] = data0[:,1]
x2_[:] = data0[:,2]
x3_[:] = data0[:,3]

#

mask_ = (x2_>=0) & (x2_<=1)

x0 = x0_[mask_]
x1 = x1_[mask_]
x2 = x2_[mask_]
x3 = x3_[mask_]
Equation = Equation_[mask_]

if (SCALE_DATASET_):
	x3 = x3 *  1.0/(1-x0) #

# ------------- #

X = np.zeros((x0.shape[0],3))
y = np.zeros((x0.shape[0]))

X[:,0] = x0[:]
X[:,1] = x1[:]
X[:,2] = x2[:]
y[:]   = x3[:]
select_k_features_case_ = 3

model = PySRRegressor.from_file(fin)

model.set_params(  extra_sympy_mappings={  "tetra": lambda x: x**4, "OpInv": lambda x: 1/x, "Lin1": lambda x: 1-x,
        "log10": lambda x: sympy.log(Abs(x),10), "PowXY": lambda x,y: Abs(x)**y })
model.refresh()

print('*** SUMMARY ***')
print(model.equations_)

srEquationMax_ = model.equations_.shape[0]
srEquation     = np.zeros((x0.shape[0],srEquationMax_))

srREabsMax = np.zeros((srEquationMax_))
srREabsMin = np.zeros((srEquationMax_))
srREMax    = np.zeros((srEquationMax_))
srREMin    = np.zeros((srEquationMax_))
srREabsMaxL20 = np.zeros((srEquationMax_))
srREabsMaxL10 = np.zeros((srEquationMax_))
srREabsMaxL5  = np.zeros((srEquationMax_))
srREabsMaxL1  = np.zeros((srEquationMax_))

eqsCount_=0
for eqsCount_ in range(0, srEquationMax_):
    srEquation[:,eqsCount_] = model.predict(X,eqsCount_)[:]
    srRE    = (srEquation[:,eqsCount_]-x3[:])/x3[:] * 100.0
    srREabs = abs((srEquation[:,eqsCount_]-x3[:])/x3[:] * 100.0  )
    srREabsMax[eqsCount_] = srREabs.max()
    srREabsMin[eqsCount_] = srREabs.min()
    srREMax[eqsCount_]    = srRE.max()
    srREMin[eqsCount_]    = srRE.min()
    srre_smaller20 = (srREabs <= 20)
    srre_smaller10 = (srREabs <= 10)
    srre_smaller5  = (srREabs <= 5)
    srre_smaller1  = (srREabs <= 1)
    srREabsMaxL20 [eqsCount_] = np.count_nonzero(srREabs[srre_smaller20])/srREabs.shape[0]*100
    srREabsMaxL10 [eqsCount_] = np.count_nonzero(srREabs[srre_smaller10])/srREabs.shape[0]*100
    srREabsMaxL5 [eqsCount_] = np.count_nonzero(srREabs[srre_smaller5])/srREabs.shape[0]*100
    srREabsMaxL1 [eqsCount_] = np.count_nonzero(srREabs[srre_smaller1])/srREabs.shape[0]*100
 

print('*** SUMMARY OF RELATIVE ERRORS ***')
dfout = pd.DataFrame({'absReMax':srREabsMax,'absReMin':srREabsMin,'ReMax':srREMax,'ReMin':srREMin, 'srREabsMaxL20':srREabsMaxL20, 'srREabsMaxL10':srREabsMaxL10, 'srREabsMaxL5':srREabsMaxL5, 'srREabsMaxL1':srREabsMaxL1 })
print (dfout)


# calculation of relative error


print('*** SUMMARY OF RELATIVE ERRORS FOR SELECTED EQUATION ***')

Equation = model.predict(X)[:]
RE = (Equation[:]-x3[:])/x3[:] * 100.0
REabs = abs((Equation[:]-x3[:])/x3[:] * 100.0  )
print('Maximum relative error %:', RE.max())
print('Minimum relative error %:', RE.min())
print('Maximum Abs Relative error %:', REabs.max())
print('Minimum Abs Relative error %:', REabs.min())
re_smaller20 = (REabs <= 20)
re_smaller10 = (REabs <= 10)
re_smaller5  = (REabs <= 5)
re_smaller1  = (REabs <= 1)
print('Abs Relative error smaller than 20%:',np.count_nonzero(REabs[re_smaller20])/REabs.shape[0]*100)
print('Abs Relative error smaller than 10%:',np.count_nonzero(REabs[re_smaller10])/REabs.shape[0]*100)
print('Abs Relative error smaller than  5%:',np.count_nonzero(REabs[re_smaller5])/REabs.shape[0]*100)
print('Abs Relative error smaller than  1%:',np.count_nonzero(REabs[re_smaller1])/REabs.shape[0]*100)
