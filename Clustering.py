"""Кластеризация

Выполнение кластеризации на K = 3 кластера методом К-средних на основании таблицы некоторых синтетических данных.

Подключение библиотек
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

"""Чтение данных: так как названия конфет уникальны, они используются в качестве индекса. Удаление столбца Cluster из набора данных."""

df = pd.read_csv("/content/data.csv", delimiter=',', index_col='Object')
df_norm = df.drop('Cluster', axis=1)

"""Инициализация модели, обучение модели на данных из df_norm. 
При выполнении задания с помощью библиотеки sklearn используется начальная инициализация со следующими координатами центроидов и параметрами:
KMeans(n_clusters=3, init=np.array([[11.8, 11.6], [8.5, 9.83], [14.0, 14.5]]), max_iter=100, n_init=1)
"""

kmeans = KMeans(n_clusters=3, init=np.array([[11.8, 11.6], [8.5, 9.83], [14.0, 14.5]]), max_iter=100, n_init=1)
model = kmeans.fit(df_norm)

"""Вывод назначенных кластеров:"""

df_norm["Clusters"] = model.labels_.tolist()
df_norm

"""Обучение модели и расчет расстояний до центроидов:"""

alldistances = kmeans.fit_transform(df_norm.drop('Clusters',axis=1))

"""вывод расстояний от данных до всех центроидов"""

alldistances

"""По результатам выполнения кластеризации определяется среднее расстояние между объектами и центроидом, отнесенных к кластеру 0."""

search = 0
Clusters_array = []
for i in range(len(df_norm['Clusters'])):
  if (df_norm['Clusters'][i+1] == search):
    Clusters_array.append(alldistances[i][search])
sum(Clusters_array)/len(Clusters_array)