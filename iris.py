
import numpy as np
seed=np.random.seed
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV #przeszukiwanie siatki parametrow

#wczytanie danych
df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/'
    'machine-learning-databases/iris/iris.data',
    header = None
)
df.columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class label']

X = df[['Sepal length', 'Sepal width','Petal length', 'Petal width']].values
#zamiana nazwy klasy na cyfry
y = pd.factorize(df['Class label'])[0]
#print(y)

#zmiana wymiaru wektora y, aby mozna bylo zadeklarowac "goraca jedynke"
yy=y.reshape(-1,1)


#deklaracja "goracej jedynki", zmiana danych tak, aby jedynka byla przyporzadkowana na miejscu klasy
enc=OneHotEncoder()
enc.fit(yy)
#transformacja z enkodera, zakodowanie numerow klas
yy_hot_full=enc.transform(yy).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, yy_hot_full, test_size=0.2, random_state=0)

#wczytanie modulu MLPClassifier ze scikit learn
mlp=MLPClassifier(hidden_layer_sizes=(50,),max_iter=1000,alpha=1e-4,solver='sgd',verbose=10,random_state=1,learning_rate_init=.1)
#ustawienie softmax
mlp.out_activation='softmax'
mlp.fit(X_train,y_train)


print("Training test score: %f" %mlp.score(X_train,y_train))
print("Test set score:%f" %mlp.score(X_test,y_test))

#przeszukiwanie siatki parametrow, aby dobrac jak najlepsze wartosci
parameters={'learning_rate_init':(0.1,0.01,0.001),'hidden_layer_sizes':(20,40,60,80,100),'solver':('adam','lbfgs','sgd')}

clf=GridSearchCV(mlp,parameters)
print(clf.fit(X,y)) #przejscie treningu klasyfikatora przez wszystkie parametry

print("Grid Search score: %f" %clf.score(X,y))
sorted(clf.cv_results_.keys())
plot_confusion_matrix(clf, X, y)
plt.show()




