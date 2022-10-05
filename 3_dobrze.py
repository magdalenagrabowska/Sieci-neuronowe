
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import numpy as np



class Adaline(object):
    """Perceptron classifier.
    
    Parameters
    ------------
    eta: float 
        Learning rate (between 0.0 and 1.0)
    n_iter: int
        Number of epochs, i.e., passes over the training dataset.
        
    Attributes
    ------------
    w_: 1d-array
        Weights after fitting.
    errors_: list
        Number of misclassifications in every epoch.
    random_state : int
        The seed of the pseudo random number generator.
    """
    
    def __init__(self, eta=0.01, n_iter=10, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : array-like; shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like; shape = [n_samples]
            Target values, or labels.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0.0
            for xi, yi in zip(X, y):
                output = self.predict2(xi) 
                error = (yi - output)
                self.w_[1:] += self.eta * error * xi
                self.w_[0] += self.eta * error
                errors += int(error != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def predict2(self,X):
        """Return class label after unit step"""
        return self.net_input(X)


#importowanie danych
iris=datasets.load_iris()
X1=iris.data[0:150,0:4] #wszystkie wartosci
yy1=iris.target[0:150]

#rozdzielanie danych; dla danej klasy, ktora wykrywac ma perceptron, przypisywane jest 1, do reszty przypisywane jest -1
def prepare_output(y,class_value):
    yy=[]
    for x in y:
        if x==class_value:
            x=1
        else:
            x=-1
        yy.append(x)
    y=yy
    return y

y1=prepare_output(yy1,0)
y2=prepare_output(yy1,1)
y3=prepare_output(yy1,2)

#print(y1)

#rozdzielanie danych na dane treningowe (80%) oraz dane testowe (20%)
X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.2,random_state=0)
X2_train,X2_test,y2_train,y2_test=train_test_split(X1,y2,test_size=0.2,random_state=0)
X3_train,X3_test,y3_train,y3_test=train_test_split(X1,y3,test_size=0.2,random_state=0)

#skalowanie danych

sc=StandardScaler()
sc.fit(X1)
X1_train_std=sc.transform(X1_train)
X1_test_std=sc.transform(X1_test)

X2_train_std=sc.transform(X2_train)
X2_test_std=sc.transform(X2_test)


X3_train_std=sc.transform(X3_train)
X3_test_std=sc.transform(X3_test)


#zdefiniowanie trzech perceptronow
ppn1=Adaline(n_iter=100,eta=0.01,random_state=0)
ppn2=Adaline(n_iter=100,eta=0.01,random_state=0)
ppn3=Adaline(n_iter=100,eta=0.01,random_state=0)

#klasyfikacja trzech klas za pomoca trzech perceptronow
ppn1.fit(X1_train_std,y1_train)
ppn2.fit(X1_train_std,y2_train)
ppn3.fit(X1_train_std,y3_train)


#predykcja dla danych treningowych
y1_train_pred=ppn1.predict(X1_train_std)
y2_train_pred=ppn2.predict(X2_train_std)
y3_train_pred=ppn3.predict(X3_train_std)

#wyswietlenie, na ile dobrze dziala klasyfikacja i ile jest zle sklasyfikowanych perceptronow
#dla danych treningowych
#print('Training data')
#print('Misclassified samples: %d' % (y1_train != y1_train_pred).sum())
#print('Accuracy: %.2f' % accuracy_score(y1_train, y1_train_pred))

#print('Misclassified samples: %d' % (y2_train != y2_train_pred).sum())
#print('Accuracy: %.2f' % accuracy_score(y2_train, y2_train_pred))

#print('Misclassified samples: %d' % (y3_train != y3_train_pred).sum())
#print('Accuracy: %.2f' % accuracy_score(y3_train, y3_train_pred))
#print()


#predykcja dla danych testowych
y1_pred=ppn1.predict(X1_test_std)
y2_pred=ppn2.predict(X2_test_std)
y3_pred=ppn3.predict(X3_test_std)

#wyswietlenie, na ile dobrze dziala klasyfikacja i ile jest zle sklasyfikowanych perceptronow
#dla danych testowych
#print('Test data')
#print('Misclassified samples: %d' % (y1_test != y1_pred).sum())
#print('Accuracy: %.2f' % accuracy_score(y1_test, y1_pred))

#print('Misclassified samples: %d' % (y2_test != y2_pred).sum())
#print('Accuracy: %.2f' % accuracy_score(y2_test, y2_pred))

#print('Misclassified samples: %d' % (y3_test != y3_pred).sum())
#print('Accuracy: %.2f' % accuracy_score(y3_test, y3_pred))
#print()

#skalowanie wszystkich danych
sc=StandardScaler()
sc.fit(X1)
X1_std=sc.transform(X1)


#predykcja za pomoca kazdego z perceptronow dla danej a
#odpowiedz z kazdego perceptronu, najwieksza wartosc oznacza przynaleznosc do danej klasy
#argmax zwraca numer przynaleznosci do danej klasy, przedzial 0-2
a=10
y1_pred0=ppn1.predict2(X1_std[a,:])
y2_pred0=ppn2.predict2(X1_std[a,:])
y3_pred0=ppn3.predict2(X1_std[a,:])
#print('Wykrycie przynależności dla 10 danej')
#print([y1_pred0,y2_pred0,y3_pred0])
#print(np.argmax([y1_pred0,y2_pred0,y3_pred0]))

#print('Wykrycie przynaleznosci do danych od 0 do 10')
#for number in range(0, 150):#pamietac o tab 
#    y1_pred0=ppn1.predict2(X1_std[number,:])
#    y2_pred0=ppn2.predict2(X1_std[number,:])
#    y3_pred0=ppn3.predict2(X1_std[number,:])
#    print([y1_pred0,y2_pred0,y3_pred0])
#    print(np.argmax([y1_pred0,y2_pred0,y3_pred0]))