from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import numpy as np


# Importacion archivos.
news = '/Users/carloslopez/Documents/Tareas Machine Learning/T3_201126513/news.txt'
labels = '/Users/carloslopez/Documents/Tareas Machine Learning/T3_201126513/labels.txt'
# Organizar Noticias y Categorias en arreglos.
with open(news) as f:
    todas_noti = f.readlines()
with open(labels) as f:
    todas_categ = f.readlines()
# Filtrar categorias fuera de la clases especificadas.
j=0
noti_filtr=[]
categ_filtr=[]
filtros=['world','sci_tech','entertainment']
for i in range (0,len(todas_categ)):
    for categ in filtros:
        if  categ in todas_categ[i]:
            noti_filtr.append([])
            categ_filtr.append([])
            noti_filtr[j]=todas_noti[i]
            categ_filtr[j]=todas_categ[i]
            j=j+1
# Asignar clases
clases = [w.replace('entertainment', '0').replace('world', '1').replace('sci_tech', '0') for w in categ_filtr]
clases = [int(c) for c in clases]
noticias = noti_filtr
# Generar bolsa palabras
vectorizer = CountVectorizer()
bolsa = vectorizer.fit_transform(noticias)
# Separar datos entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(bolsa, clases ,test_size=0.1,)
# Crear Modelo
errores=[]
gammas=[]
ces=[]
pos=0
C_rbf=[0.01, 0.1, 1, 10, 100]
gamma_rbf=[0.001, 0.01, 0.1, 1, 10]
'''
for Cs in C_rbf:
    for gmms in gamma_rbf:
'''
svr_rbf = SVR(kernel='rbf',C=10 , gamma=0.01)
modelo = svr_rbf.fit(x_train, y_train)
# Prueba
pred=modelo.predict(x_test)
# Se pasa el resultado por una funcion de activacion de escalon
pred = [round(c) for c in pred]
#Calculo error
count=0
for j in range(len(pred)):
    if pred[j]==y_test[j]:
        count = count+1

err = count/len(pred)
print (err)

'''
        errores.append([])
        errores[pos]= err*1000
        gammas.append([])
        gammas[pos]= gmms
        ces.append([])
        ces[pos]= Cs

        pos=pos+1

np.savetxt('Error.txt', errores, fmt='%d')
np.savetxt('Gamma.txt', gammas, fmt='%d')
np.savetxt('Cs.txt', ces, fmt='%d')
'''
