import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.cluster import KMeans

# Завантажуємо набір даних
iris_df = datasets.load_iris()

# Описуємо модель
model = KMeans(n_clusters=3)

# Проводимо моделювання
model.fit(iris_df.data)

# Передбачення на одиничному прикладі
predicted_label = model.predict([[7.2, 3.5, 0.8, 1.6]])

# Передбачення на всьому наборі даних
all_predictions = model.predict(iris_df.data)

# Виводимо результати передбачення
print(predicted_label)
print(all_predictions)

# Використовуємо PCA для зменшення розмірності даних до 2D
pca = PCA(n_components=2)
iris_2d = pca.fit_transform(iris_df.data)

# Додаємо прогнозовані мітки до даних
iris_df['predicted_label'] = all_predictions

# Створюємо scatter plot для візуалізації
plt.figure(figsize=(10, 6))
sns.scatterplot(x=iris_2d[:, 0], y=iris_2d[:, 1], hue=iris_df['predicted_label'], palette='Set1', s=100)

# Виводимо центроїди
centers_2d = pca.transform(model.cluster_centers_)
plt.scatter(centers_2d[:, 0], centers_2d[:, 1], marker='X', s=200, c='red', label='Centroids')

plt.title('KMeans Clustering of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()
