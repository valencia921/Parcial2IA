import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

dataset = pd.read_csv('C:\\Users\\Usuario\\Desktop\\Semestre 9\\Inteligencia Artificial\\Clases\\Parcial 2\\Nacimientos.csv')
print(dataset.describe())


dataset.plot(x='Peso en gramos', y='Talla centimetros', style='o')
plt.title('Peso en gramos vs Talla centimetros')
plt.xlabel('Peso en gramos')
plt.ylabel('Talla centimetros')
plt.show()

x = dataset['Peso en gramos'].values.reshape(-1,1)
y = dataset['Talla centimetros'].values.reshape(-1,1)

#--------Codigo de minimos cuadrados----------------

auxSumatoriaX = 0
auxSumatoriaY = 0
auxSumatoriaXY =0
auxSumatoriaX2= 0
lista1= dataset['Peso en gramos'].values
lista2 = dataset['Talla centimetros'].values
cant = len(lista1)

auxSumatoriaX = sum(lista1)
auxSumatoriaY = sum(lista2)

def obtenerMult(lista1,lista2):
  suma=0
  for i in range(len(lista1)):
    suma += lista1[i] * lista2[i]
  return suma

auxSumatoriaXY = obtenerMult(lista1,lista2)

def obtenerPotencia(lista1):
  suma=0
  for i in range(len(lista1)):
    suma+= lista1[i]**2
  return suma

auxSumatoriaX2 = obtenerPotencia(lista1)

minimo_cuadrado = (auxSumatoriaXY - (auxSumatoriaX * auxSumatoriaY)/ cant) / (auxSumatoriaX2 - (auxSumatoriaX**2)/cant)
print(f'--------------------------------------------------')
print(f'------------Algoritmo creado----------------------')

print(f'Pendiente (m)= {minimo_cuadrado}')

b = (auxSumatoriaY/cant) - (minimo_cuadrado * (auxSumatoriaX /cant) )
print(f'Punto de corte (b)=  {b}')

print(f'--------------------------------------------------')



#---------------------------------------------------



#test_size es nos permite definir  la proporciÃ³n del conjunto de pruebas.
from sklearn.linear_model import LinearRegression # importo solo la regresiÃ³n lineal
r_l=LinearRegression() # creamos una instancia del modelo lineal

r_l.fit(x, y)
print(f'---------------Linear Regression-------------------')
print('m = '+str(r_l.coef_)+ ', b = ' + str(r_l.intercept_))
print('Ecuación dada: y='+str(r_l.coef_)+'x +'+str(r_l.intercept_))
y_pred=r_l.predict(x.reshape(-1,1))
plt.scatter(x, y)
plt.plot(x,y_pred,  color='blue')
plt.plot(x, y, color='green')
plt.title('Peso en gramos vs Talla centimetros')
plt.xlabel('Peso en gramos')
plt.ylabel('Talla centimetros')
plt.show()

from sklearn.metrics import mean_squared_error, r2_score

print (f'Error cuadratico medio: %.2f' % mean_squared_error(y, y_pred))
print (f'Estadistico R^2 aprox: %.2f' % r2_score(y, y_pred))
r2 = r_l.score(x.reshape(-1,1),y)
print(f'R^2 {r2}')

print(f'--------------------------------------------------')

# Metodo de Gradiente descendiente
print(f'------Gradiente descendiente------------')


# función para calcular el error
# busca valores de b y m (hipótesis)que minimice el error devuelto por esta función
def coste(x, y, b, m):
  n = len(x)
  error = 0.0
  for i in range(n):
    h = b + m * x[i]
    error += (y[i] - h) ** 2
  return error / (2 * n)


def descenso_gradiente(x, y, b, m, alpha, epochs):
  n = len(x)
  hist_coste = []
  for ep in range(epochs):
    m_deriv = 0
    b_deriv = 0
    for i in range(n):
      h = b + m * x[i]
      b_deriv += h - y[i]
      m_deriv += (h - y[i]) * x[i]
      hist_coste.append(coste(x, y, b, m))
    b -= (b_deriv / n) * alpha
    m -= (m_deriv / n) * alpha

  return b, m, hist_coste


b = b
m = minimo_cuadrado
alpha = 0.00001
iters = 50
b, m, hist_coste = descenso_gradiente(x, y, b, m, alpha, iters)


print(f'Punto de corte : {b}\n',
      f'Pendiente : {m}')

y_pre = b + m * x

plt.scatter(x, y)
plt.plot(x, y_pre, 'red')
plt.title('Peso en gramos vs Talla centimetros')
plt.xlabel('Peso en gramos')
plt.ylabel('Talla centimetros')
plt.show()

plt.plot(hist_coste)
plt.title('Peso en gramos vs Talla centimetros')
plt.xlabel('Peso en gramos')
plt.ylabel('Talla centimetros')
plt.show()

print(f'----------------------------------------------------------')

#-----------------------Regresión Logística---------------

print(f'------Regresion Logistica------------')

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

y = (dataset['Ficha de información publicación datos abiertos'] == 'MASCULINO').astype(np.int_)

#objeto para estimar el clasificador logistico
log_reg = LogisticRegression()

#Con los datos extraidos de la base de datos entreno
log_reg.fit(x, y)

#Estimo las probabilidades de naciemiento que sean maculino con un peso
# que varían de 1.9 a 3.99 kg, a este nuevo se le da el nombre de X_new
X_new = np.linspace(1.9, 3.99, 500).reshape(-1, 1)
#solo es la prediccion
y_proba1 = log_reg.predict(X_new)
print("Datos de la predicción ",y_proba1)
#y_proba es el elemento de la probabilidad segun los datos
y_proba = log_reg.predict_proba(X_new)
print("Datos de la predicción ",y_proba)
plt.plot(X_new, y_proba[:, 1], "g-", label= "MASCULINO")
plt.plot(X_new, y_proba[:, 0], "b--", label = "FEMENINO")
plt.title('Probalidad segun los datos de Genero y Peso')
plt.xlabel('Peso en gramos')
plt.ylabel('Probabilidad')
plt.show()

print(f'--------------------------------------------------')

#---------------------KNN------------------------------------

print(f'---------------------KNN------------------------------------')

#Se importan la librerias a utilizar
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model #para el modelo lineal
from sklearn.metrics import mean_squared_error, r2_score # para generar el error cuadratico y el R^2

#Pequeño analisis descriptivo del dataset
print('Características del dataset:')
print(dataset)

from sklearn.model_selection import train_test_split

#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Defino el algoritmo a utilizar
from sklearn.neighbors import KNeighborsClassifier

algoritmo = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p = 2)
#Entreno el modelo
algoritmo.fit(X_train, y_train)
#Realizo una predicción
y_pred = algoritmo.predict(X_test)

#Calculo la precisión del modelo
from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred)
print('Precisión del modelo:')
print(precision)
print(f'---------------------------------------------------------')

#---------------------Regresión Ridge------------------------------------

print(f'-------------------Regresión Ridge----------------------------------')
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model #para el modelo lineal

x = dataset['Peso en gramos'].values.reshape(-1,1)
y = dataset['Talla centimetros'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(x, y)
from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train, y_train)
print("Datos con Regresion Ridge")
print("Estadístico R^2, con los datos de entrenamiento: {:.2f}".format(ridge.score(X_train, y_train)))
print("Estadístico R^2, con los datos de test {:.2f} \n".format(ridge.score(X_test, y_test)))
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train, y_train)
print("Datos con Regresion Lineal")
print("Estadístico R^2, con los datos de entrenamiento: {:.2f}".format(lr.score(X_train, y_train)))
print("Estadístico R^2, con los datos de test {:.2f}\n".format(lr.score(X_test, y_test)))

#si se varia alpha
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("Datos con Regresion Ridge con alpha igual a 10")
print("Estadístico R^2, con los datos de entrenamiento: {:.2f}".format(ridge10.score(X_train, y_train)))
print("Estadístico R^2, con los datos de test: {:.2f}\n".format(ridge10.score(X_test, y_test)))

ridge0 = Ridge(alpha=0.1).fit(X_train, y_train)
print("Datos con Regresion Ridge con alpha igual a 0.1")
print("Estadístico R^2, con los datos de entrenamiento: {:.2f}".format(ridge0.score(X_train, y_train)))
print("Estadístico R^2, con los datos de test: {:.2f}".format(ridge0.score(X_test, y_test)))


#---------------------Regresión Polinomica------------------------------------

print(f'-------------------Regresión Polinomica----------------------------------')

# Importamos la clase de Regresión Lineal de scikit-learn
from sklearn.linear_model import LinearRegression
# para generar características polinómicas
from sklearn.preprocessing import PolynomialFeatures
import numpy as np #Librería para vectores y matrices
import pandas as pd

pf = PolynomialFeatures(degree = 3)    # usaremos polinomios de grado 3
dataset = pd.read_csv('C:\\Users\\Usuario\\Desktop\\Semestre 9\\Inteligencia Artificial\\Clases\\Parcial 2\\Nacimientos.csv')
x = dataset['Peso en gramos'].values.reshape(-1,1)
y = dataset['Talla centimetros'].values.reshape(-1,1)

X = pf.fit_transform(x.reshape(-1,1))  # transformamos la entrada en polinómica
regresion_lineal = LinearRegression() # creamos una instancia de LinearRegression
# instruimos a la regresión lineal que aprenda de los datos (ahora polinómicos) (X,y)
regresion_lineal.fit(X, y)
# vemos los parámetros que ha estimado la regresión lineal
print('w = ' + str(regresion_lineal.coef_) + ', b = ' + str(regresion_lineal.intercept_))
# resultado: w = [0.         -2.76365236  0.02574253  0.79923842], b = -1.5873146347130387

from sklearn.metrics import mean_squared_error # importamos el cálculo del error cuadrático medio (MSE)
# Predecimos los valores y para los datos usados en el entrenamiento
prediccion_entrenamiento = regresion_lineal.predict(X)
# Calculamos el Error Cuadrático Medio (MSE = Mean Squared Error)
mse = mean_squared_error(y_true = y, y_pred = prediccion_entrenamiento)
# La raíz cuadrada del MSE es el RMSE
rmse = np.sqrt(mse)
print('Error Cuadrático Medio (MSE) = ' + str(mse))
print('Raíz del Error Cuadrático Medio (RMSE) = ' + str(rmse))
# calculamos el coeficiente de determinación R2
r2 = regresion_lineal.score(X, y)
print('Coeficiente de Determinación R2 = ' + str(r2))
print('------------------------------------------------------------------------------')