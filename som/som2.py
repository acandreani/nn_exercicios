import matplotlib.pylab as plt
# import sompy as sompy
import pandas as pd
import numpy as np
from time import time
import sompy


def ex1():

	filename = "iris.data"

	dataset = pd.read_csv(filename, header=None, names=['sepal_length','sepal_width','petal_length','petal_width','species'])
	dataset.head()


	Data1=dataset.columns[0:4]

	from sklearn.preprocessing import LabelBinarizer

	species_lb = LabelBinarizer()

	labels=dataset.species.values
	type_labels,numeric_labels=np.unique(labels, return_inverse=True)
	print(numeric_labels)

	Y = species_lb.fit_transform(dataset.species.values)

	from sklearn.preprocessing import normalize

	FEATURES = dataset.columns[0:4]
	X_data = dataset[FEATURES].as_matrix()
	X_data = normalize(X_data)


	# fig = plt.figure()
	# plt.plot(Data1[:,0],Data1[:,1],'ob',alpha=0.2, markersize=4)
	# fig.set_size_inches(7,7)
	# plt.show()

	mapsize = [20,20]
	som = sompy.SOMFactory.build(X_data, mapsize, mask=None, mapshape='planar', lattice='rect',
	 normalization='var', initialization='pca', neighborhood='gaussian',
	  training='batch', name='sompy')  # this will use the default parameters, but i can change the initialization and neighborhood methods

	numeric_str_labels=list(map(str,numeric_labels))

	print(numeric_str_labels)

	som.data_labels=np.array(numeric_str_labels)

	som.train(n_job=1, verbose='info')  # verbose='debug' will print more, and verbose=None wont print anything


	u = sompy.umatrix.UMatrixView(50, 50, 'umatrix', 
		show_axis=False, text_size=8, show_text=True)

	#This is the Umat value
	UMAT  = u.build_u_matrix(som, distance=1, row_normalized=False)

	#Here you have Umatrix plus its render
	UMAT = u.show(som, distance2=1, row_normalized=False, 
		show_data=False, contooor=False, blob=False, labels=numeric_str_labels)


def ex2():
		filename = "iris.data"

	dataset = pd.read_csv(filename, header=None, names=['sepal_length','sepal_width','petal_length','petal_width','species'])
	dataset.head()


	Data1=dataset.columns[0:4]

	from sklearn.preprocessing import LabelBinarizer

	species_lb = LabelBinarizer()

	labels=dataset.species.values
	type_labels,numeric_labels=np.unique(labels, return_inverse=True)
	print(numeric_labels)

	Y = species_lb.fit_transform(dataset.species.values)

	from sklearn.preprocessing import normalize

	FEATURES = dataset.columns[0:4]
	X_data = dataset[FEATURES].as_matrix()
	X_data = normalize(X_data)


	# fig = plt.figure()
	# plt.plot(Data1[:,0],Data1[:,1],'ob',alpha=0.2, markersize=4)
	# fig.set_size_inches(7,7)
	# plt.show()

	mapsize = [20,20]
	som = sompy.SOMFactory.build(X_data, mapsize, mask=None, mapshape='planar', lattice='rect',
	 normalization='var', initialization='pca', neighborhood='gaussian',
	  training='batch', name='sompy')  # this will use the default parameters, but i can change the initialization and neighborhood methods

	numeric_str_labels=list(map(str,numeric_labels))

	print(numeric_str_labels)

	som.data_labels=np.array(numeric_str_labels)

	som.train(n_job=1, verbose='info')  # verbose='debug' will print more, and verbose=None wont print anything


	u = sompy.umatrix.UMatrixView(50, 50, 'umatrix', 
		show_axis=False, text_size=8, show_text=True)

	#This is the Umat value
	UMAT  = u.build_u_matrix(som, distance=1, row_normalized=False)

	#Here you have Umatrix plus its render
	UMAT = u.show(som, distance2=1, row_normalized=False, 
		show_data=False, contooor=False, blob=False, labels=numeric_str_labels)


if __name__=="__main__":
	ex1()
	ex2()



