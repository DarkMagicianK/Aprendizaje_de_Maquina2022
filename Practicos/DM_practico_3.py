import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import norm
import os
sns.set_theme(style="darkgrid")

#%%
os.system("cls")
from sklearn.datasets import load_iris
iris = load_iris();
print ("Atributos")
print (iris.feature_names)

x1_index = 0;
x2_index = 1;
X = iris.data;
x1 = X[:,x1_index];
x2 = X[:,x2_index];

#%%
# Plots 2D entre dos features
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
plt.figure(figsize=(5,4));
plt.scatter(x1,x2,c=iris.target,cmap="viridis");
plt.colorbar(ticks=[0, 1, 2],format=formatter)
#plt.colorbar(ticks=[1, 2, 3])
#plt.set_yticklabels(['< -1', '0', '> 1'])  # vertically oriented colorbar
plt.xlabel(iris.feature_names[x1_index])
plt.ylabel(iris.feature_names[x2_index])

#%% Primer test
# Instansiacion del Modelo
from sklearn import neighbors
param_k = 1;
X, y_target = X[:,(x1_index,x2_index)], iris.target;
knn1 = neighbors.KNeighborsClassifier(n_neighbors=param_k);
knn1.fit(X, y_target);
# Hacemos una predicción
x_new = [[3, 5]];
#print(iris.target_names[knn1.predict([[3, 5, 4, 2]])]);
print(iris.target_names[knn1.predict(x_new)]);
plt.scatter(x1,x2,c=iris.target,cmap="viridis");
plt.colorbar(ticks=[0, 1, 2],format=formatter)
plt.scatter(x_new[0][0],x_new[0][1],c="red",marker="x");

#%% Segundo test
param_k = 3;# Hiper parametro
pts = 100;   # para armar la grilla
X, y_target = X[:,(x1_index,x2_index)], iris.target;
knn2 = neighbors.KNeighborsClassifier(n_neighbors=param_k);
knn2.fit(X, y_target);

#% Predicción Mesh
x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, pts),
                     np.linspace(y_min, y_max, pts));
Z = knn2.predict(np.c_[xx.ravel(), yy.ravel()])

# Plot also the training points
from matplotlib.colors import ListedColormap
# Create color maps for 3-class classification problem, as with iris
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z,cmap=cmap_light)
plt.scatter(x1,x2,c=iris.target,cmap=cmap_bold);
plt.colorbar(ticks=[0, 1, 2],format=formatter)

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
#plt.axis('tight')

# Hacemos una predicción
x_new = [[3, 5]];
plt.scatter(x_new[0][0],x_new[0][1],c="black",marker="x");

#%%

#%%
os.system ('cls')
print(os.name)

#%%
from sklearn.inspection import DecisionBoundaryDisplay
_,ax = plt.subplots();
DecisionBoundaryDisplay.from_estimator(
        knn2, X,
        cmap=cmap_light, alpha=0.8, ax=ax,
        response_method="predict",
        plot_method="pcolormesh",
        shading="auto");

#%%
import sklearn
print(phyton.__version__)