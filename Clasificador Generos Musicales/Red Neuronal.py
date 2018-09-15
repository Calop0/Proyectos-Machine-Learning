import csv
import numpy as np
from matplotlib import pyplot as plt
from numpy import genfromtxt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Crear Arreglo con los datos del csv
datos = genfromtxt('msd_genre_dataset2.csv', delimiter=',') #dataset1(1:0) dataset2(1:-1)
df_x=datos[:,1:]
df_y=datos[:,0]

#Separar datos en sets de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.1)
#Preprocesamiento!!
scaler = StandardScaler()
scaler.fit(x_train)
#Grupo 1 de Datos (10% del total)
in1 = 0
fin1 = in1 + round(0.1*len(x_train))
x_train1 = x_train[in1:fin1,:]
y_train1 = y_train[in1:fin1]
#Grupo 2 de Datos (20% del total)
in2 = fin1 + 1
fin2 = in2 + round(0.2*len(x_train))
x_train2 = x_train[in2:fin2,:]
y_train2 = y_train[in2:fin2]
#Grupo 3 de Datos (70% del total)
in3 = fin2 + 1
fin3 = len(x_train)
x_train3=x_train[in3:fin3,:]
y_train3=y_train[in3:fin3]

##--------------------------- METODO FUERZA BRUTA ------------------------------
#Inicializacion Variables
min=(0,0)
err_min=1
resultados=np.zeros(200)
capa1=np.zeros(200)
capa2=np.zeros(200)
pos=0
#cX indica el numero de neuronas en la capa X
for c1 in range (1,101,5):
    for c2 in range(1,101,5):
        #Sintonizacion de Red Neuronal Grupo 2
        red_neuronal = MLPClassifier(activation='logistic',solver='lbfgs',hidden_layer_sizes=(c1,c2),learning_rate_init=0.0001, alpha=1, max_iter=1000)
        red_neuronal.fit(x_train1,y_train1)
        pred=red_neuronal.predict(x_test)
        #Calculo Error
        count=0
        for i in range(len(pred)):
            if pred[i]==y_test[i]:
                count=count+1
                err=1-count/len(pred)

        if  err<err_min:
            err_min=err #Actualiza menor error encontrado
            min=(c1,c2) #Actualiza combinacion del menor error

        #Crea vectores con resultados y numero de neuronas en cada capa
        pos=pos+1
        resultados[pos]=err*10000
        capa1[pos]=c1
        capa2[pos]=c2

print(min)
print(err_min)


np.savetxt('resultados3_10.txt', resultados, fmt='%d')
np.savetxt('capa1_10.txt', capa1, fmt='%d')
np.savetxt('capa2_3.txt', capa2, fmt='%d')

'''
##--------------------------- METODO MANUAL ------------------------------------
c1=300
c2=200
c3=100
c4=50

#Sintonizacion de Red Neuronal Grupo 0 (Todos los datos)
red_neuronal0 = MLPClassifier(activation='logistic',solver='adam',hidden_layer_sizes=(c1),learning_rate_init=0.0001, alpha=1, max_iter=1000)
red_neuronal0.fit(x_train3,y_train3)
pred0=red_neuronal0.predict(x_test)
#Calculo Error
count0=0
for i in range(len(pred0)):
    if pred0[i] == y_test[i]:
        count0 = count0+1
err0 = 1-count0/len(pred0)

print (err0)
'''
