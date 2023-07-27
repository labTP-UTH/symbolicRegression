import time
import numpy as np
import pandas as pd
from pysr import *
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
import sympy
from sympy import*
import numpy, sys, getopt


start = time.time()

# preliminaries
SCALE_DATASET_     = true
PARTITION_DATASET_ = true


# input dataset
df0 = pd.read_csv('ShortTubeData.csv')
data0 = numpy.loadtxt( 'ShortTubeData.csv', skiprows = 1, delimiter=',', dtype=float ); print ("Total instances from input file (total data)  = " , data0.shape[0])
print ("Total features  = " , data0.shape[1])
print("\n*End reading input dataset -> continue with symbolic regression...*\n")

#

x0 = np.zeros((df0.shape[0]))
x1 = np.zeros((df0.shape[0]))
x2 = np.zeros((df0.shape[0]))
x3 = np.zeros((df0.shape[0]))
x0[:] = data0[:,0]
x1[:] = data0[:,1]
x2[:] = data0[:,2]
x3[:] = data0[:,3]


# pysr parameters

default_pysr_params_ = dict(
    niterations=10,
    populations=3*12,
    population_size=50,
    ncyclesperiteration=500,
    weight_randomize=0.03,
    adaptive_parsimony_scaling=100,
    weight_optimize=0.01,
    parsimony=1.0e-5,
    loss="L2DistLoss()",
    precision=64,
    warm_start=True,
    maxsize=35,
    maxdepth=10,
##    warmup_maxsize_by=0.2,
##    batching=True, batch_size=1000,
    binary_operators=["+", "*", "-", #"^", #"/",
        "PowXY(x,y)=abs(x)^y"],
    unary_operators=["square","cube", "tetra(x)=x^4",
        "exp", "log10",
        "OpInv(x)=1.0f0/x",
        "Lin1(x)=1.0f0-x"
                    ],
    constraints={
        "PowXY":(9,2),
        "square":1,
        "cube":1,
        "tetra":1
                },
    extra_sympy_mappings={ "tetra": lambda x: x**4, "OpInv": lambda x: 1/x, "Lin1": lambda x: 1-x, "log10": lambda x: sympy.log(Abs(x),10), "PowXY": lambda x,y: Abs(x)**y },
    nested_constraints={
        "Lin1"    : {"Lin1":0, "exp":1, "log10":1, "PowXY":1, "OpInv": 0, "square":1, "cube":1, "tetra":1 },
        "exp"     : {"Lin1":0, "exp":0, "log10":0, "PowXY":1, "OpInv": 0, "square":1, "cube":1, "tetra":1 },
        "log10"   : {"Lin1":0, "exp":0, "log10":0, "PowXY":0, "OpInv": 0, "square":1, "cube":1, "tetra":1 },
        "PowXY"   : {"Lin1":0, "exp":0, "log10":0, "PowXY":0, "OpInv": 0, "square":1, "cube":1, "tetra":1 },
        "OpInv"   : {"Lin1":1, "exp":1, "log10":1, "PowXY":1, "OpInv": 0, "square":1, "cube":1, "tetra":1 },
        "square"  : {"Lin1":0, "exp":0, "log10":0, "PowXY":0, "OpInv": 0, "square":0, "cube":0, "tetra":0 },
        "cube"    : {"Lin1":0, "exp":0, "log10":0, "PowXY":0, "OpInv": 0, "square":0, "cube":0, "tetra":0 },
        "tetra"   : {"Lin1":0, "exp":0, "log10":0, "PowXY":0, "OpInv": 0, "square":0, "cube":0, "tetra":0 }
                      },
    complexity_of_operators={"log10":2, "exp":2, "Lin1":2, "PowXY":2, "OpInv":2, "square":2, "cube":2, "tetra":2},
    complexity_of_constants=2,
    complexity_of_variables=1

)



# pysr model


model = PySRRegressor(
    procs=12,
    timeout_in_seconds=2 * 60 * 60 * 24,
    early_stop_condition=(
        "stop_if(loss, complexity) = loss < 1e-8 && complexity < 10"), # Stop early if we find a good and simple equation
    **default_pysr_params_
                      )

######################################################################################

pardata = 0
pardatamax_ = 4
while pardata < pardatamax_:

    print ('partition data= ', pardata)
    mask_ = (x2>=0) & (x2<=1)
    elem_skip_  = 4
    elem_start_ = pardata

    x0Mod_ = x0[mask_]
    x1Mod_ = x1[mask_]
    x2Mod_ = x2[mask_]
    x3Mod_ = x3[mask_]
    
    if (SCALE_DATASET_):
    	x3Mod_ = x3Mod_ * 1.0/(1-x0Mod_)
    
    if (PARTITION_DATASET_):
    	x0Mod_ = x0Mod_[elem_start_::elem_skip_]
    	x1Mod_ = x1Mod_[elem_start_::elem_skip_]
    	x2Mod_ = x2Mod_[elem_start_::elem_skip_]
    	x3Mod_ = x3Mod_[elem_start_::elem_skip_]

    X = np.zeros((x0Mod_.shape[0],3))
    y = np.zeros((x0Mod_.shape[0]))
    X[:,0] = x0Mod_[:]
    X[:,1] = x1Mod_[:]
    X[:,2] = x2Mod_[:]
    y[:]   = x3Mod_[:]

    model.fit(X, y)
    
    pardata += 1

######################################################################################

print(model)

x1 = np.linspace(0,2,100)
y1 = x1

plt.plot(x1, y1, '-r', label='y=x')
plt.scatter(y[:], model.predict(X))
plt.xlabel('Truth')
plt.ylabel('Prediction')
plt.show()
plt.close()

r_squared = model.score(X, y)
print('r2 score', r_squared)

f = model.predict(X)
r2 = r2_score(y, f)
print('r2 score (2nd est)', r2)

print("\nLatex:")
print(model.latex(index=None, precision=8))

print(model.latex_table(indices=None, precision=8, columns=['equation', 'complexity', 'loss', 'score']))

print("\nSympy:")
sympy_model =model.sympy(index=None)
print(sympy_model)

end = time.time()
print("The time of execution is :", (end-start) , "s")
