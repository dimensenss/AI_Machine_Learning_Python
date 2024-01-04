import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE


# Завантаження даних ірисів Фішера
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Використання t-SNE для зменшення розмірності до 3D
tsne = TSNE(n_components=3, random_state=42)
X_tsne = tsne.fit_transform(X)

# Створення 3D графіку
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Розфарбовуємо точки відповідно до класів
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=y, cmap=plt.cm.get_cmap("viridis"), marker='o')

# Додаємо легенду
legend = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend)

# Додаємо мітки до вісей
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D Visualization')

# Показуємо графік
plt.show()
