import numpy as np 
import pandas as pd 

from sklearn.preprocessing import MinMaxScaler
import math
from minisom import MiniSom
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error

#obliczanie odleglosci pomiedzy wektorami
def eu_dist(t1, t2):
    return ((t1[0] - t2[0])**2 + (t1[1] - t2[1])**2+ (t1[2] - t2[2])**2 +(t1[3] - t2[3])**2+ (t1[4] - t2[4])**2)**0.5

#wczytanie wartosci
data = pd.read_csv('/home/pcet/poczatki_py/sn_3/all_stocks_5yr.csv')
cl = data[data['Name']=='LRCX']
X_df = cl[['close','low','high','volume','open']]
y_df=cl[['open']]

#skalowanie danych w zakresie od 0 do 1, aby otrzymać poprawnie wyskalowane wyniki
scaler = MinMaxScaler(feature_range = (0,1))
X1=scaler.fit_transform(X_df)
y1=scaler.fit_transform(y_df)
som_shape=(5,1)
X_train=X1[0:1248]
y_train=y1[1:1249]
#X_test=X1[1249:1253]

#algorytm Kohonena, wyznaczenie centroidow
som = MiniSom(5, 1, 5, sigma=0.5,neighborhood_function='gaussian',activation_distance='euclidean', learning_rate=0.7)
som.pca_weights_init(X_train)
som.train(X_train, 1000, verbose=True)
winner_coordinates = np.array([som.winner(x) for x in X_train]).T
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)

#wyznaczenie r, najdluzsza odlegosc pomiedzy wektorami
r=0
for i in range(0,X_train.shape[0]):
    for j in range(0,X_train.shape[0]):
        if(eu_dist(X_train[j],X_train[i])>r):
            r=eu_dist(X_train[j],X_train[i])


#wyznaczenie elementow macierzy
dist=np.empty(X_train.shape[0]*5,dtype=float)
licz=0
for i in range(0,X_train.shape[0]):
    tmp=0
    fmp=0
    for c in som.get_weights():
        c=c.ravel()
        tmp=eu_dist(c, X_train[i])*math.exp((-(eu_dist(c, X_train[i])/r))**2) #odleglosc pomiedzy wektorem a centroidem ponmnożona przez fi
        dist[licz]=tmp
        licz=licz+1

print(X_train.shape[0])   
dist=dist.reshape(X_train.shape[0],X_train.shape[1]) #macierz fi

p=np.linalg.pinv(dist) #pseudoodwrotność macierzy
#print(p.shape)

#obliczenie wag w sieci
w=p.dot(y_train) #funkcja mnożenia macierzy przez wektor
#print(w)

#wyznaczenie wartosci przewidywanych
y_pred=np.empty(X_train.shape[0],dtype=float)
licz2=0
for i in range(0,X_train.shape[0]):
    tmp=0
    fmp=0
    licz=0
    for c in som.get_weights():
        c=c.ravel()
        tmp=w[licz]*eu_dist(c, X_train[i])*math.exp((-(eu_dist(c, X_train[i])/r))**2)
        fmp=fmp+tmp
        licz=licz+1
    y_pred[licz2]=fmp
    licz2=licz2+1


print("Aglebraiczna sieć RBF")
print("Max error: ")
print(max_error(y_train, y_pred))

print("Variance score: ")
print(explained_variance_score(y_train, y_pred))
