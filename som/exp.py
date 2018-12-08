from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
from som import SOM
from sklearn.datasets import load_iris

data = load_iris()
flower_data = data['data']
normed_flower_data = flower_data / flower_data.max(axis=0)
target_int = data['target']
target_names = data['target_names']
targets = [target_names[i] for i in target_int]

#Train a 20x30 SOM with 400 iterations
som = SOM(25, 25, 4, 100) # My parameters
som.train(normed_flower_data)

#Get output grid
image_grid = som.get_centroids()
print(image_grid)

#Map colours to their closest neurons
mapped = som.map_vects(normed_flower_data)

#Plot
plt.imshow(image_grid)
plt.title('SOM')
for i, m in enumerate(mapped):
    plt.text(m[1], m[0], targets[i], ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.5, lw=0))
plt.show() 