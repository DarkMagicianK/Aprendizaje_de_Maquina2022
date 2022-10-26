import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import norm
sns.set_theme(style="darkgrid")

#%% Ejercicio N°1 - Casos covid
# -----------------------------------------------------------------------------
def ajuste_norm(data,ax):
    mu, std = norm.fit(data);
    xmin,xmax = ax.get_xlim();
    x = np.linspace(xmin, xmax, 50);
    p = norm.pdf(x, mu, std);
    ax.plot(x, p, 'k', linewidth=2)
    
#%%

#url_file = 'https://github.com/manlio99/Materia-de-aprendizaje/blob/master/4_DataWrangling/data/casos_covid_bahia.csv'
url_file_casos = 'https://raw.githubusercontent.com/manlio99/Materia-de-aprendizaje/master/4_DataWrangling/data/casos_covid_bahia.csv';

data_covid = pd.read_csv(url_file_casos);
orden = data_covid.shape;
nro_cols = orden[1];
namecols = data_covid.columns;
print(data_covid);

# Plots de cols respecto al tiempo
_, ax1 = plt.subplots(2,5, figsize = ( 20, 8));
ax_row = -1;
for i in range(10):
  if (i%5 ==0): ax_row = ax_row +1
  sns.lineplot(x="fecha", y=namecols[i+1],data=data_covid, ax=ax1[ax_row][i%5])

# Histogramas de cols
_, ax2 = plt.subplots(2,5, figsize = ( 20, 8))
ax_row = -1;
for i in range(10):
  if (i%5 ==0): ax_row = ax_row +1
  sns.histplot(x=namecols[i+1],data=data_covid, bins=10,stat="density", ax=ax2[ax_row][i%5])

# Ajute normal
ax_row = -1;
for i in range(9):
  if (i%5 ==0): ax_row = ax_row +1
  ajuste_norm(data_covid[namecols[i+1]],ax2[ax_row][i%5])  

sns.pairplot(data_covid,diag_kind="kde")  
#%% Respuestas
# a) Según los histogramas obtenidos, ninguna variable presenta indicios de una distribución normal sobre su soporte.
# Podria decirse que la variable, *descartados* presenta una tendencia a una distribucion uniforme.

# b)
# Comportamiento sospechoso: La variable,
# *contencion_psicologica*: presenta dos cluster definidos. Puede deberse a la saturación del equipo de trabajo.
# *monitoreo_epidemiologico*: presenta dos cluster definidos tambien. En epoca de crecemiento de casos de covid, puede que la baja a la mitad practicamente se deba a una politica ministerial. En uno de los clusters se exhibe un comportamiento gausiano.
#%%

#%% Ejercicio N°1 -Camas covid
url_file_camas = 'https://raw.githubusercontent.com/manlio99/Materia-de-aprendizaje/master/4_DataWrangling/data/camas_covid_bahia.csv';
data_camas = pd.read_csv(url_file_camas);
namecols = data_camas.columns;
print(data_camas);

del data_camas[namecols[0]];
namecols = data_camas.columns;
orden = data_camas.shape;
nro_cols = orden[1];
#
# Plots de cols respecto al tiempo
_, ax1 = plt.subplots(4,4, figsize = ( 20, 14));
ax_row = -1;
for i in range(nro_cols-2):
  if (i%4 ==0): ax_row = ax_row +1
  sns.lineplot(x="fecha", y=namecols[i+1],data=data_camas, ax=ax1[ax_row][i%4])

#
# Histogramas de cols
_, ax2 = plt.subplots(4,4, figsize = ( 20, 14))
ax_row = -1;
for i in range(nro_cols-2):
  if (i%4 ==0): ax_row = ax_row +1
  sns.histplot(x=namecols[i+1],data=data_camas, bins=10,stat='density', ax=ax2[ax_row][i%4])  

# Ajute normal
ax_row = -1;
for i in range(nro_cols-2):
  if (i%4 ==0): ax_row = ax_row +1
  ajuste_norm(data_camas[namecols[i+1]],ax2[ax_row][i%4])  

sns.pairplot(data_camas,diag_kind="kde")  
  
  
#%% Respuestas
# a) Según los histogramas obtenidos, "camas_ocupadas_hospitales","camas_sospechosos_covid", presentan a priori una distribución normal sobre su soporte(descartando los outliers por datos faltantes).
# Podria decirse que la variable, *descartados* presenta una tendencia a una distribucion uniforme.

# b)

#%% Ejercicio N°2 - 
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme(style="darkgrid")

# Funciones adicionales
def clasificador(datos,T):
    prediccion = np.zeros(datos.shape);
    prediccion[datos > T] = 1;
    return(prediccion);

def indicadores(clase,prediccion,N):
    # np.sum(prediccion): cant. total de detecciones + (suma predicciones positivas))
    # N:                  cant. total casos correctos (suma de condiciones positivas)

    TPv = np.logical_and(clase,prediccion);
    TP = np.sum(TPv);
    Precision = TP/(np.sum(prediccion)); #TP / (cant. total de detecciones + (suma predicciones positivas))
    Sensibilidad = TP/(N);
    return((Precision,Sensibilidad))    
    
# Generacón de Datos Sinteticos
N = 1000; # Cantidad de datos
[muA, sigmaA] = [0, 1]; # media y desvio estandar
[muB, sigmaB] = [2, 1]; # media y desvio estandar
datosA = np.random.normal(muA, sigmaA, size=(N,1)); #creando muestra de datosA
datosB = np.random.normal(muB, sigmaB, N)[:,None]; #creando muestra de datosB
claseA = np.zeros(datosA.shape);
claseB = np.ones(datosB.shape);
Datos = np.concatenate((datosA,datosB));
Clases = np.concatenate( (claseA,claseB) );
Predicciones = np.zeros(Clases.shape);
#%%
# Graficando histograma
# histograma de distribución normal.
_,ax = plt.subplots(1,2)
#sns.histplot(data=datosA, bins=10, ax=ax[0])
_,ax1 = plt.subplots(1,2)
countA,xA = np.histogram(datosA);
countB,xB = np.histogram(datosB);
ax1[0].plot(xA[0:len(xA)-1],countA);
ax1[0].plot(xB[0:len(xB)-1],countB);

# Clasificación e indicadores
T_vec = np.linspace(Datos.min(), Datos.max(),30);
Precision = np.zeros(T_vec.shape);
Sensibilidad = np.zeros(T_vec.shape);
  
for i in range(T_vec.size):
    T = T_vec[i];
    Predicciones = clasificador(Datos,T);
    # Indicadores
    pys = indicadores(Clases,Predicciones,N); # N es  ant. total casos correctos (suma de condiciones positivas)
    Precision[i] = pys[0];
    Sensibilidad[i] = pys[1];

ax1[1].plot(Sensibilidad,Precision);
#ax1[1].set_ylim([-0,1.1])
ax1[1].grid(alpha=0.5)
ax[1].set_title('Precisión vs Recall')


#%%
sns.pairplot(data_covid,diag_kind="kde")