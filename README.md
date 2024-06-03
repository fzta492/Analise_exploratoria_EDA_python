
### Análise de Dados Exploratória (EDA) em Python

#### Contexto
A Análise de Dados Exploratória (EDA) é um passo crucial na análise de dados. Ela envolve várias técnicas e métodos para resumir as características principais dos dados, muitas vezes com métodos visuais. A EDA é utilizada para detectar padrões, identificar anomalias, testar hipóteses e verificar suposições.

#### Perguntas Comuns de Análise Exploratória
Antes de começar a EDA, é importante ter em mente algumas perguntas que guiarão a análise:
- Quais são os valores médios, medianos e de moda das variáveis numéricas?
- Quais são as distribuições das variáveis categóricas?
- Existem outliers nos dados? Como eles influenciam a análise?
- Quais variáveis estão correlacionadas entre si?
- Existem padrões ou tendências sazonais nos dados?
- Como as diferentes variáveis interagem entre si?
- Existem agrupamentos naturais ou padrões nos dados?

### Etapas da EDA com Exemplos de Código

#### 1. Definir Objetivos da Análise
Este passo é teórico e não requer código específico. O entendimento do problema e as perguntas a responder são cruciais.

#### 2. Coletar e Carregar os Dados
- **Obtenção dos Dados**: Reunir dados de diferentes fontes, como bancos de dados, arquivos CSV, etc.
- **Carregamento dos Dados**: Importar os dados para um ambiente de análise (como Python, R, ou softwares de BI).

```python
import pandas as pd

# Carregar dados de um arquivo CSV
df = pd.read_csv('seu_arquivo.csv')

# Mostrar as primeiras linhas do dataframe
print(df.head())
```

#### 3. Limpeza dos Dados
- **Tratar Valores Ausentes**: Identificar e tratar dados faltantes (por exemplo, imputação ou exclusão).
- **Correção de Erros**: Identificar e corrigir inconsistências e erros nos dados.
- **Remover Duplicatas**: Identificar e remover registros duplicados.
- **Conversão de Tipos de Dados**: Certificar-se de que cada coluna tem o tipo de dado correto.

```python
# Tratar valores ausentes
df = df.dropna()  # Remove linhas com valores ausentes
# ou
df = df.fillna(df.mean())  # Preenche valores ausentes com a média da coluna

# Correção de erros e conversão de tipos de dados
df['coluna'] = df['coluna'].astype('tipo_desejado')

# Remover duplicatas
df = df.drop_duplicates()

# Mostrar informações do dataframe
print(df.info())
```

#### 4. Análise Estatística Descritiva
- **Sumarização Estatística**: Calcular estatísticas descritivas (média, mediana, moda, desvio padrão, etc.).
- **Distribuição de Frequência**: Analisar a distribuição de frequências para variáveis categóricas.

```python
# Sumarização estatística
print(df.describe())

# Distribuição de frequência para variáveis categóricas
print(df['coluna_categórica'].value_counts())
```

#### 5. Visualização de Dados
- **Histogramas**: Visualizar a distribuição de variáveis numéricas.
- **Gráficos de Barras**: Analisar a distribuição de variáveis categóricas.
- **Box Plots**: Detectar outliers e entender a dispersão dos dados.
- **Gráficos de Dispersão**: Explorar relações entre variáveis numéricas.
- **Gráficos de Correlação**: Visualizar a correlação entre diferentes variáveis (matriz de correlação).

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Histograma
df['coluna_numérica'].hist()
plt.show()

# Gráfico de barras
df['coluna_categórica'].value_counts().plot(kind='bar')
plt.show()

# Box plot
sns.boxplot(x=df['coluna_categórica'], y=df['coluna_numérica'])
plt.show()

# Gráfico de dispersão
sns.scatterplot(x=df['coluna1'], y=df['coluna2'])
plt.show()

# Matriz de correlação
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()
```

#### 6. Detecção e Tratamento de Outliers
- **Identificação de Outliers**: Usar técnicas como box plots e Z-score para identificar outliers.
- **Tratamento de Outliers**: Decidir sobre a remoção, transformação ou manutenção de outliers.

```python
# Box plot para identificar outliers
sns.boxplot(x=df['coluna_numérica'])
plt.show()

# Remover outliers usando Z-score
from scipy import stats
import numpy as np

z_scores = np.abs(stats.zscore(df['coluna_numérica']))
df = df[(z_scores < 3)]
```

#### 7. Análise de Correlação
- **Matriz de Correlação**: Calcular a correlação entre variáveis para identificar relações.
- **Mapas de Calor**: Visualizar a matriz de correlação usando um mapa de calor.

```python
# Matriz de correlação
corr = df.corr()
print(corr)

# Mapa de calor
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()
```

#### 8. Transformação de Dados
- **Normalização/Padronização**: Ajustar a escala dos dados.
- **Transformações Logarítmicas**: Aplicar transformações para lidar com distribuições não normais.
- **Codificação de Variáveis Categóricas**: Transformar variáveis categóricas em numéricas (one-hot encoding, label encoding).

```python
# Normalização
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[['coluna_numérica']] = scaler.fit_transform(df[['coluna_numérica']])

# Transformação logarítmica
df['coluna_numérica'] = np.log(df['coluna_numérica'] + 1)

# Codificação de variáveis categóricas
df = pd.get_dummies(df, columns=['coluna_categórica'])
```

#### 9. Agrupamento de Dados (Opcional)
- **Clusterização**: Usar técnicas como K-means para identificar grupos nos dados.
- **Análise de Componentes Principais (PCA)**: Reduzir a dimensionalidade dos dados mantendo a maior variância possível.

```python
# Clusterização usando K-means
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
df['cluster'] = kmeans.fit_predict(df[['coluna1', 'coluna2']])

# PCA para redução de dimensionalidade
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df[['coluna1', 'coluna2']])
```

#### 10. Interpretação e Documentação
- **Relatórios**: Documentar as descobertas e insights obtidos durante a EDA.
- **Conclusões**: Fazer recomendações baseadas nas análises realizadas.
- **Visualizações**: Criar gráficos e tabelas que resumam os principais insights.

