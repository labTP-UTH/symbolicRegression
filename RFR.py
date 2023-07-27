import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

data=pd.read_csv('MData.csv', sep = ',', header=0, names=[r'$p$',r'$l$',r'$\delta$',r'$W$'])

in_vars=[r'$p$',r'$l$',r'$\delta$']
X1=data[in_vars]
y= data[r'$W$']

scaler = StandardScaler()
scaler.fit(X1)
Xsc = scaler.transform(X1)
X = pd.DataFrame(Xsc, columns= in_vars)

corr = data.corr()
ax = sns.heatmap(
    corr, 
    annot=True,
    linewidth=0.5,
    fmt=".1f",
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    horizontalalignment='right'
);
ax.set_yticklabels(
    ax.get_yticklabels(),
    horizontalalignment='right'
);


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.80,
                                   test_size=.20, random_state=10)


model =  RandomForestRegressor(n_estimators = 30, random_state = 42)  #7
model.fit(X_train, y_train)
y_pred_tr = model.predict(X_train)
y_pred_te = model.predict(X_test)

##Identity and residuals plot
f, ((ax1, ax2)) = plt.subplots(1, 2,figsize=(10, 4))
p1 = max(max(y_train), max(y_test))
p2 = 0
ax1.scatter(y_train, y_pred_tr, s=2,c='navy', label='train data')
ax1.scatter(y_test, y_pred_te, s=2, c='darksalmon', label='test data')
ax1.set_xlabel(r'$W$', fontsize=12)
ax1.set_ylabel(r'$W_{p}^{RFR}$', fontsize=12)
ax1.plot([p1, p2], [p1, p2], 'r-')
rr=model.score(X_test, y_test)
ax1.set_xlim([0, 1.5])
ax1.set_ylim([0, 1.5])
ax1.legend()
plt.grid(True)

from yellowbrick.regressor import ResidualsPlot
visualizer = ResidualsPlot(model, ax=ax2, train_color='navy', test_color='darksalmon',title=' ',  classes=["train", "test"])#['train data','test data'] ) #, hist=False, qqplot=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
ax2.set_xlim([0, 1.5])
ax2.set_ylim([-0.08, 0.08])
ax2.legend(fontsize=10)
visualizer.show()

#The feature importance plot
from yellowbrick.model_selection import FeatureImportances
f = plt.subplots(figsize=(4, 3))
viz = FeatureImportances(model,relative=False, colors=['y','darkgrey','darksalmon'], title=' ', 
                         xlabel=r'$Feature$'+ ' ' + r'$Importance$')
viz.fit(X_train, y_train)
plt.tight_layout()
viz.finalize()
