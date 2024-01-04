import seaborn as sns
from sklearn import datasets
import pandas as pd


# Загружаємо дані ірисів
iris_data = datasets.load_iris()
iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
iris_df['species'] = iris_data.target_names[iris_data.target]

# Застосовано власні маркери
sns.pairplot(iris_df, hue="species", palette="husl", markers=["o", "o", "o"])

# Видаляємо верхні та праві краї графіку
sns.despine()

# Показуємо графік
import matplotlib.pyplot as plt
plt.show()
