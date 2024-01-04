from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


iris = load_iris()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(iris.data)

dbscan = DBSCAN(eps=0.67, min_samples=5)
dbscan.fit(X_scaled)
labels = dbscan.labels_

plt.scatter(X_scaled[:,0], X_scaled[:,1], c=labels)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('DBSCAN clusters')
plt.show()
