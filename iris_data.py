from minisom import MiniSom
from sklearn import preprocessing
import numpy as np
import sklearn.datasets
import pandas as pd
import math
import matplotlib.pyplot as plt

#inicjalizacja danych, skalowanie
iris = sklearn.datasets.load_iris()
data = iris.data[:, :4]
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data1 = np.array(min_max_scaler.fit_transform(data)) #dane wejsciowe, dostepne jest tylko przeskalowane X
start=0.7 #poczatkowy parametr wspolczynnika uczenia

##PARAMETRY-WSPOLCZYNNIKI UCZENIA
#Siec SOM - liniowe zmniejszanie wspolczynnika uczenia
   
current_learning_rate=(start*(100-20))/100
som = MiniSom(1, 3, 4, sigma=0.5,neighborhood_function='gaussian',activation_distance='euclidean', learning_rate=current_learning_rate)
som.pca_weights_init(data1)
# Train the SOM network with 1000 epochs
som.train(data1, 1000, verbose=True)

som_shape=(1,3)
# each neuron represents a cluster
winner_coordinates = np.array([som.winner(x) for x in data1]).T
# with np.ravel_multi_index we convert the bidimensional
# coordinates to a monodimensional index
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
# plotting the clusters using the first 2 dimentions of the data
#print(cluster_index.shape)
#print(data1.shape)
for c in np.unique(cluster_index):
    plt.scatter(data1[cluster_index == c, 0],
                data1[cluster_index == c, 1], label='cluster='+str(c), alpha=.7)

# plotting centroids
#for centroid in som.get_weights():
#    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', 
#              s=10, linewidths=20, color='k', label='centroid')
plt.title('liniowe zmniejszanie wspolczynnika uczenia')
plt.legend()
plt.show()

#Siec SOM- wykladnicze zmniejszanie wspolczynnika uczenia
C=-0.04
current_learning_rate=start*math.exp(C*20)
som = MiniSom(3, 1, 4, sigma=1.0,neighborhood_function='gaussian',learning_rate=current_learning_rate)
som.pca_weights_init(data)
 #Train the SOM network with 1000 epochs
som.train(data, 1000, verbose=True)
som_shape=(3,1)
# each neuron represents a cluster
winner_coordinates = np.array([som.winner(x) for x in data]).T
# with np.ravel_multi_index we convert the bidimensional
# coordinates to a monodimensional index
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
# plotting the clusters using the first 2 dimentions of the data
for c in np.unique(cluster_index):
    plt.scatter(data[cluster_index == c, 0],
                data[cluster_index == c, 1], label='cluster='+str(c), alpha=.7)#

# plotting centroids
#for centroid in som.get_weights():
#    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', 
#                s=10, linewidths=20, color='k', label='centroid')

plt.title('wykladnicze zmniejszanie wspolczynnika uczenia')
plt.legend()
plt.show()


#Siec SOM-hiperboliczne zmniejszanie wspolczynnika uczenia
#for rate in range(1,100):   
C1=1
C2=1
current_learning_rate=C1/(C2+20)
som = MiniSom(3, 1, 4, sigma=1.0,neighborhood_function='gaussian', learning_rate=current_learning_rate)
som.pca_weights_init(data)
# Train the SOM network with 1000 epochs
som.train(data, 1000, verbose=True)
som_shape=(3,1)
    # each neuron represents a cluster
winner_coordinates = np.array([som.winner(x) for x in data]).T
# with np.ravel_multi_index we convert the bidimensional
#coordinates to a monodimensional index
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
# plotting the clusters using the first 2 dimentions of the data
for c in np.unique(cluster_index):
    plt.scatter(data[cluster_index == c, 0],
                data[cluster_index == c, 1], label='cluster='+str(c), alpha=.7)

# plotting centroids
#for centroid in som.get_weights():
#    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', 
#                s=10, linewidths=20, color='k', label='centroid')
plt.title('hiperboliczne zmniejszanie wspolczynnika uczenia')
plt.legend()
plt.show()

##PARAMETRY-NORMY
#norma Euklidesowa
som = MiniSom(1, 3, 4, sigma=0.5,activation_distance='euclidean', learning_rate=0.5)
som.pca_weights_init(data)
# Train the SOM network with 1000 epochs
som.train(data, 1000, verbose=True)

som_shape=(1,3)
# each neuron represents a cluster
winner_coordinates = np.array([som.winner(x) for x in data]).T
# with np.ravel_multi_index we convert the bidimensional
# coordinates to a monodimensional index
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
# plotting the clusters using the first 2 dimentions of the data
for c in np.unique(cluster_index):
    plt.scatter(data[cluster_index == c, 0],
                data[cluster_index == c, 1], label='cluster='+str(c), alpha=.7)

# plotting centroids
#for centroid in som.get_weights():
#    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', 
#                s=10, linewidths=20, color='k', label='centroid')
plt.title('norma Euklidesowa')
plt.legend()
plt.show()


#norma skalarna - cosine
som = MiniSom(1, 3, 4, sigma=0.5,activation_distance='cosine', learning_rate=0.5)
som.pca_weights_init(data1)
## Train the SOM network with 1000 epochs
som.train(data1, 1000, verbose=True)

som_shape=(1,3)
# each neuron represents a cluster
winner_coordinates = np.array([som.winner(x) for x in data]).T
# with np.ravel_multi_index we convert the bidimensional
# coordinates to a monodimensional index
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
# plotting the clusters using the first 2 dimentions of the data
for c in np.unique(cluster_index):
    plt.scatter(data[cluster_index == c, 0],
                data[cluster_index == c, 1], label='cluster='+str(c), alpha=.7)

# plotting centroids
#for centroid in som.get_weights():
#    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', 
#                s=10, linewidths=20, color='k', label='centroid')
plt.title('norma skalarna ')
plt.legend()
plt.show()

#norma bezwzględna - manhattan (taksówkowa)
som = MiniSom(1, 3, 4, sigma=0.5,activation_distance='manhattan', learning_rate=0.5)
som.pca_weights_init(data)
## Train the SOM network with 1000 epochs
som.train(data, 1000, verbose=True)

som_shape=(1,3)
# each neuron represents a cluster
winner_coordinates = np.array([som.winner(x) for x in data]).T
# with np.ravel_multi_index we convert the bidimensional
# coordinates to a monodimensional index
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
# plotting the clusters using the first 2 dimentions of the data
for c in np.unique(cluster_index):
    plt.scatter(data[cluster_index == c, 0],
                data[cluster_index == c, 1], label='cluster='+str(c), alpha=.7)

# plotting centroids
#for centroid in som.get_weights():
#    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', 
#                s=10, linewidths=20, color='k', label='centroid')
plt.title('Norma bezwzgledna')
plt.legend()
plt.show()