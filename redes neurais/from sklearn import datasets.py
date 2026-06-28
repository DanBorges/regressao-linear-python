from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

iris = datasets.load_iris()

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(iris.data, iris.target, test_size = 0.3, random_state=0)

modelo = MLPClassifier(
  verbose=True,
  hidden_layer_sizes=(5,4),
  activation='relu',
  batch_size=20,
  learning_rate='adaptive',
  momentum=0.9,
  early_stopping=False,
  max_iter=1000,
  random_state=10
    
)
modelo.fit(X_treinamento, y_treinamento)

plt.plot(modelo.loss_curve_)
plt.xlabel("Iterações")
plt.ylabel("Valor de Loss")
plt.title("Loss")
plt.show()

previsoes = modelo.predict(X_teste)

print(accuracy_score(y_teste, previsoes))