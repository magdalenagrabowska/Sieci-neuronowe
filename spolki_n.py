import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('/home/pcet/poczatki_py/sn_3/all_stocks_5yr.csv')


cl = data[data['Name']=='LRCX']
X_df = cl[['close','low','high','volume','open']]

def preprocessData(data,wyjscie, k):
    X,Y = [],[]
    for i in range(len(data)-k-1):
        x_i_mat=np.array(data[i:(i+k)])
        x_i = x_i_mat.reshape(x_i_mat.shape[0]*x_i_mat.shape[1])
        y_i= np.array(data[(i+k):(i+k+1)][wyjscie])
        X.append(x_i) #X jest zbiorem wszystkich wartosci z ostatnich 5 dni
        Y.append(y_i) #y jest zbiorem, ktory chcemy przewidywac
    return np.array(X),np.array(Y)

X,y = preprocessData(X_df, 'open',k=5)
print(X.shape)
#skalowanie danych w zakresie od 0 do 1, aby otrzymać poprawnie wyskalowane wyniki
scaler = MinMaxScaler(feature_range = (0,1))
X1=scaler.fit_transform(X)
yy=scaler.fit_transform(y)
#zmiana y w wektor (tablice jednowymiarowa) 
y1=np.ravel(yy)


print(X1.shape)
print(y1.shape)

X_train=X1[0:1248]
X_test=X1[1249:1253]
y_train=y1[0:1248]
y_test=y1[1249:1253]
#podzielenie X oraz y na zbior danych treningowych oraz testowych
#X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=0)
#print(X_train[1::5]) #bierze co piata tablice, zaczynajac od drugiej


#X_test1=np.reshape(X_test,(1255,5))
#for i in range(len(X_test1)):
#    print(X_test1[i,1])

#stworzenie modelu preceptronu wielowarstwowego do zadania regresji
#regr = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', learning_rate='constant', learning_rate_init=0.001, max_iter=500, shuffle=True, validation_fraction=0.1,epsilon=1e-08,n_iter_no_change=10)
#regr.fit(X_train, y_train)

#expected_y=y_test
#predicted_y=regr.predict(X_test)
#predicted_y1=np.reshape(predicted_y,(-1,1))
#print(expected_y)
#print(predicted_y)
#print(expected_y.shape)
#print(predicted_y.shape)

#x=np.linspace(0,251,num=251)
#plt.plot(x,expected_y,x,predicted_y)
#plt.legend(['expected y','predicted y'])
#plt.show()

#for i in range(len(X_test1)):
#   print(X_test1[i,1])
#
# print(expected_y)
#print(predicted_y)


#print("Training test score: %f" %regr.score(X_train,y_train))
#print("Test set score:%f" %regr.score(X_test,y_test))

#y1_test=np.reshape(y_test,(-1,1))
#y1_train=np.reshape(y_train,(-1,1))


#obliczenie pierwistka błędu średniokwadratowego
#trainPredict = regr.predict(X_train)
#testPredict = regr.predict(X_test)
#trainPredict=np.reshape(trainPredict,(-1,1))
#testPredict =np.reshape(testPredict ,(-1,1))
# invert predictions
#trainPredict = scaler.inverse_transform(trainPredict)
#trainY = scaler.inverse_transform([y_train])
#testPredict = scaler.inverse_transform(testPredict)
#testY = scaler.inverse_transform([y_test])
# calculate root mean squared error
#print()
#print('Calculate root mean squared error')
#trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
#print('Train Score: %.2f RMSE' % (trainScore))
#testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
#print('Test Score: %.2f RMSE' % (testScore))

#przeszukiwanie siatki parametrow, aby dobrac jak najlepsze wartosci
#parameters={'learning_rate_init':(0.1,0.01,0.001),'hidden_layer_sizes':(20,40,60,80,100),'solver':('adam','lbfgs','sgd'),'activation':('tanh','logistic'),'learning_rate':('constant','adaptive')}

#clf=GridSearchCV(regr,parameters)
#print(clf.fit(X1,y1)) #przejscie treningu klasyfikatora przez wszystkie parametry

#print('Best parameters')
#clf.refit=1
#print(clf.best_params_)

#print("Grid Search score: %f" %clf.score(X1,y1))