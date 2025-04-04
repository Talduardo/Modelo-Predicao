# Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from backend import dataset as data


# Carregando o conjunto de dados
# Substitua 'seu_arquivo.csv' pelo nome do seu arquivo
data = pd.read_csv('dataset.csv')

# Analisando os dados
data.head()

# Verificando informações basicas
data.info()

# Analisar a distribuição das variaveis
# Histogramas
data.hist(bins=30, figsize=(15,10))
plt.show()

# Correlação entre as variáveis
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()

# Preparação dos dados
data.isnull().sum()
# Preencher valores ausentes com a média, mediana ou outra estratégia adequada

#  Construção do Modelo de Regressão Linear:
# Dividir os dados em variáveis independentes (X) e dependentes (y)
X = data[['Área do lote', 'Ano de construção', 'Área do primeiro andar', 'Área do segundo andar', 'Número de banheiros completos', 'Número de quartos acima do solo', 'Número total de quartos acima do solo']]
y = data['Preço de venda da casa']

# Dividir os dados em conjuntos de treino e teste:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo:
model = LinearRegression()
model.fit(X_train, y_train)

#  Avaliação do Modelo:
# Fazer previsões:
y_pred = model.predict(X_test)

# Calcular o erro médio quadrático:
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Visualizar os resultados:
plt.scatter(y_test, y_pred)
plt.xlabel("Preço de venda real")
plt.ylabel("Preço de venda previsto")
plt.title("Comparação entre preços reais e previstos")
plt.show()