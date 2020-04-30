from sklearn.datasets import fetch_openml

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255.

X_train, X_test = X[:2000], X[2000:]
y_train, y_test = y[:2000], y[2000:]

print(X_train)

from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

model = TSNE(n_components=2, random_state=0, perplexity=100)
tsne_data = model.fit_transform(X_train)

# creating a new data fram which help us in ploting the result data
tsne_data = np.vstack((tsne_data.T, y_train)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))

# Ploting the result of tsne
sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.title('With perplexity = 100')
plt.show()