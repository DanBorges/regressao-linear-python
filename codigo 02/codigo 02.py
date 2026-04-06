import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import scipy.stats as stats
import seaborn as sns


base = pd.read_csv('mt_cars.csv')
base.shape
base = base.drop(['Unnamed: 0'], axis=1)
#print(base.head())

corr = base.corr()
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f')
#plt.show()

column_pairs = [('mpg', 'cyl'),('mpg', 'disp'),('mpg', 'hp'),('mpg', 'wt'),('mpg', 'drat'),('mpg', 'vs')]
n_plots = len(column_pairs)
fig, axes = plt.subplots(nrows=n_plots,ncols = 1, figsize=(6,4 * n_plots))

for i, pair in enumerate(column_pairs):
    x_col, y_col = pair
    sns.scatterplot(x=x_col, y=y_col, data=base, ax=axes[i])
    axes[i].set_title(f'{x_col} vs {y_col}')

#plt.tight_layout
#plt.savefig("teste")

modelo = sm.ols(formula = 'mpg ~ wt + disp + hp', data=base) #usa variavel dependente antes da virgula
modelo = modelo.fit() # de fato cria o modelo
print(modelo.summary()) #AIC e BIC são métricas de regressão, quanto menor melhor

residuos = modelo.resid
plt.hist(residuos, bins=20)
plt.xlabel("Residuos")
plt.ylabel("Frquencia")
plt.title("Histograma de Residuos")
plt.savefig("residuos")


stats.probplot(residuos, dist="norm", plot=plt)
plt.title("Q-Q Plot de Residuos")
plt.show()


#h0 - Dados normalmente distribuidos
#p< 0.05 rejeito a hipótese nula, (Não estão normalmente distribuidos)
#p>0.05  não é possível rejeitar a hipótese nula h0
stat, pval = stats.shapiro(residuos)
print (f'Shapiro-Wilk statística: {stat:.3f}, p-value: {pval:.3f}')
