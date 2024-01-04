from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

# Завантажуємо дані ірисів
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Додаємо інформацію про вид ірису
iris_df['iris_variety'] = iris.target

# Вилучаємо виміри як масив NumPy
iris_samples = iris_df.iloc[:, :-1].values

# Реалізація ієрархічної кластеризації за допомогою функції linkage
mergings = linkage(iris_samples, method='complete')

# Будуємо дендрограму, вказавши параметри, що зручні для відображення
dendrogram(mergings,
           labels=iris.target_names[iris.target],
           leaf_rotation=90,
           leaf_font_size=6,
           color_threshold=4
           )

plt.show()
