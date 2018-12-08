import numpy as np
import matplotlib.pyplot as plt
from matplotlib  import cm
from mpl_toolkits.mplot3d import Axes3D
import somoclu
import pandas as pd
from collections import OrderedDict
def ex1_1():
	"""Experimento de referência com dataset iris"""
	n_rows, n_columns = 20, 20

	filename = "iris.data"

	dataset = pd.read_csv(filename, header=None, names=['sepal_length','sepal_width','petal_length','petal_width','species'])
	dataset.head()

	from sklearn.preprocessing import LabelBinarizer

	species_lb = LabelBinarizer()

	labels=dataset.species.values
	type_labels, numeric_labels=np.unique(labels, return_inverse=True)
	print(type_labels, numeric_labels)

	labels_legend=[]
	for i,label in enumerate(type_labels):
		labels_legend.append(tuple([str(i),label]))
	# print(numeric_labels)
	#str_labels= [ "c"+str(e) for e in numeric_labels]

	Y = species_lb.fit_transform(dataset.species.values)

	from sklearn.preprocessing import normalize

	FEATURES = dataset.columns[0:4]
	x_data = dataset[FEATURES].as_matrix()
	x_data = normalize(x_data)

	som = somoclu.Somoclu(n_columns, n_rows, compactsupport=False,neighborhood="gaussian")
	som.train(x_data,radiuscooling="linear",scalecooling="linear")


	som.view_umatrix(bestmatches=False,labels=numeric_labels,colorbar=True,labels_legend=labels_legend)

def ex1_2():
	"""variação do decaimento da função de vizinhança: exponencial"""


	n_rows, n_columns = 20, 20

	filename = "iris.data"

	dataset = pd.read_csv(filename, header=None, names=['sepal_length','sepal_width','petal_length','petal_width','species'])
	dataset.head()

	from sklearn.preprocessing import LabelBinarizer

	species_lb = LabelBinarizer()

	labels=dataset.species.values
	type_labels, numeric_labels=np.unique(labels, return_inverse=True)
	print(type_labels, numeric_labels)

	labels_legend=[]
	for i,label in enumerate(type_labels):
		labels_legend.append(tuple([str(i),label]))
	# print(numeric_labels)
	str_labels= [ "c"+str(e) for e in numeric_labels]

	Y = species_lb.fit_transform(dataset.species.values)

	from sklearn.preprocessing import normalize

	FEATURES = dataset.columns[0:4]
	x_data = dataset[FEATURES].as_matrix()
	x_data = normalize(x_data)

	som = somoclu.Somoclu(n_columns, n_rows, compactsupport=False,neighborhood="gaussian")
	som.train(x_data,radiuscooling="exponential")

	som.view_umatrix(bestmatches=False,labels=numeric_labels,colorbar=True,
		labels_legend=labels_legend)

def ex1_3():
	"""variação da função de vizinhança: bubble"""


	n_rows, n_columns = 20, 20

	filename = "iris.data"

	dataset = pd.read_csv(filename, header=None, names=['sepal_length','sepal_width','petal_length','petal_width','species'])
	dataset.head()

	from sklearn.preprocessing import LabelBinarizer

	species_lb = LabelBinarizer()

	labels=dataset.species.values
	type_labels, numeric_labels=np.unique(labels, return_inverse=True)
	print(type_labels, numeric_labels)

	labels_legend=[]
	for i,label in enumerate(type_labels):
		labels_legend.append(tuple([str(i),label]))
	# print(numeric_labels)
	str_labels= [ "c"+str(e) for e in numeric_labels]

	Y = species_lb.fit_transform(dataset.species.values)

	from sklearn.preprocessing import normalize

	FEATURES = dataset.columns[0:4]
	x_data = dataset[FEATURES].as_matrix()
	x_data = normalize(x_data)

	som = somoclu.Somoclu(n_columns, n_rows, compactsupport=False,neighborhood="bubble")
	som.train(x_data,radiuscooling="linear",scalecooling="linear",scale0=0.1)

	som.view_umatrix(bestmatches=False,labels=numeric_labels,colorbar=True,
		labels_legend=labels_legend)

def ex1_4():
	"""variação do decaimento da taxa de aprendizagem"""


	n_rows, n_columns = 20, 20

	filename = "iris.data"

	dataset = pd.read_csv(filename, header=None, names=['sepal_length','sepal_width','petal_length','petal_width','species'])
	dataset.head()

	from sklearn.preprocessing import LabelBinarizer

	species_lb = LabelBinarizer()

	labels=dataset.species.values
	type_labels, numeric_labels=np.unique(labels, return_inverse=True)
	print(type_labels, numeric_labels)

	labels_legend=[]
	for i,label in enumerate(type_labels):
		labels_legend.append(tuple([str(i),label]))
	# print(numeric_labels)
	str_labels= [ "c"+str(e) for e in numeric_labels]

	Y = species_lb.fit_transform(dataset.species.values)

	from sklearn.preprocessing import normalize

	FEATURES = dataset.columns[0:4]
	x_data = dataset[FEATURES].as_matrix()
	x_data = normalize(x_data)

	som = somoclu.Somoclu(n_columns, n_rows, compactsupport=False,neighborhood="gaussian")
	som.train(x_data,radiuscooling="linear",scalecooling="exponential",scale0=0.1)

	som.view_umatrix(bestmatches=False,labels=numeric_labels,colorbar=True,
		labels_legend=labels_legend)


def ex1_5():
	"""variação da taxa de aprendizagem:0.2"""


	n_rows, n_columns = 20, 20

	filename = "iris.data"

	dataset = pd.read_csv(filename, header=None, names=['sepal_length','sepal_width','petal_length','petal_width','species'])
	dataset.head()

	from sklearn.preprocessing import LabelBinarizer

	species_lb = LabelBinarizer()

	labels=dataset.species.values
	type_labels, numeric_labels=np.unique(labels, return_inverse=True)
	print(type_labels, numeric_labels)

	labels_legend=[]
	for i,label in enumerate(type_labels):
		labels_legend.append(tuple([str(i),label]))
	# print(numeric_labels)
	str_labels= [ "c"+str(e) for e in numeric_labels]

	Y = species_lb.fit_transform(dataset.species.values)

	from sklearn.preprocessing import normalize

	FEATURES = dataset.columns[0:4]
	x_data = dataset[FEATURES].as_matrix()
	x_data = normalize(x_data)

	som = somoclu.Somoclu(n_columns, n_rows, compactsupport=False,neighborhood="gaussian")
	som.train(x_data,radiuscooling="linear",scalecooling="linear",scale0=0.2)

	som.view_umatrix(bestmatches=False,labels=numeric_labels,colorbar=True,
		labels_legend=labels_legend)




def ex2_1():

	n_rows, n_columns = 20, 20

	filename = "isolet1_2_3_4.data"

	names=[]
	for i in range(1,618):
	    names.append("f"+str(i))

	names.append("letter_class")

	dataset = pd.read_csv(filename, header=None, names=names)
	dataset.head()

	labels=dataset.letter_class.values
	colormap=cm.ScalarMappable(cmap="tab20b")


	# seaborn.pairplot(dataset,hue="species", height=2, diag_kind="kde")
	# plt.show()
	from sklearn.preprocessing import normalize

	type_labels, numeric_labels=np.unique(labels, return_inverse=True)
	# print(numeric_labels)
	str_labels= [ "c"+str(e) for e in numeric_labels]

	print("type_labels",type_labels)
	labels_legend=[]
	for i,label in enumerate(type_labels):
		labels_legend.append(tuple([str(i),str(int(label))]))
	print(labels_legend)


	FEATURES = dataset.columns[0:617]
	x_data = dataset[FEATURES].as_matrix()
	x_data = normalize(x_data)


	som = somoclu.Somoclu(n_columns, n_rows, compactsupport=False)
	som.train(x_data)


	som.view_umatrix(bestmatches=False,labels=numeric_labels, colorbar=True)

def ex2_2():
	"""variação do decaimento da função de vizinhança: exponencial"""

	n_rows, n_columns = 20, 20

	filename = "isolet1_2_3_4.data"

	names=[]
	for i in range(1,618):
	    names.append("f"+str(i))

	names.append("letter_class")

	dataset = pd.read_csv(filename, header=None, names=names)
	dataset.head()

	labels=dataset.letter_class.values
	colormap=cm.ScalarMappable(cmap="tab20b")


	# seaborn.pairplot(dataset,hue="species", height=2, diag_kind="kde")
	# plt.show()
	from sklearn.preprocessing import normalize

	type_labels, numeric_labels=np.unique(labels, return_inverse=True)
	# print(numeric_labels)
	str_labels= [ "c"+str(e) for e in numeric_labels]

	print("type_labels",type_labels)
	labels_legend=[]
	for i,label in enumerate(type_labels):
		labels_legend.append(tuple([str(i),str(int(label))]))
	print(labels_legend)


	FEATURES = dataset.columns[0:617]
	x_data = dataset[FEATURES].as_matrix()
	x_data = normalize(x_data)


	som = somoclu.Somoclu(n_columns, n_rows, compactsupport=False,neighborhood="gaussian")
	som.train(x_data,radiuscooling="exponential")

	som.view_umatrix(bestmatches=False,labels=numeric_labels,colorbar=True)

def ex2_3():
	"""variação da função de vizinhança: bubble"""

	n_rows, n_columns = 20, 20

	filename = "isolet1_2_3_4.data"

	names=[]
	for i in range(1,618):
	    names.append("f"+str(i))

	names.append("letter_class")

	dataset = pd.read_csv(filename, header=None, names=names)
	dataset.head()

	labels=dataset.letter_class.values
	colormap=cm.ScalarMappable(cmap="tab20b")


	# seaborn.pairplot(dataset,hue="species", height=2, diag_kind="kde")
	# plt.show()
	from sklearn.preprocessing import normalize

	type_labels, numeric_labels=np.unique(labels, return_inverse=True)
	# print(numeric_labels)
	str_labels= [ "c"+str(e) for e in numeric_labels]

	print("type_labels",type_labels)
	labels_legend=[]
	for i,label in enumerate(type_labels):
		labels_legend.append(tuple([str(i),str(int(label))]))
	print(labels_legend)


	FEATURES = dataset.columns[0:617]
	x_data = dataset[FEATURES].as_matrix()
	x_data = normalize(x_data)


	som = somoclu.Somoclu(n_columns, n_rows, compactsupport=False,neighborhood="bubble")
	som.train(x_data,radiuscooling="linear",scalecooling="linear",scale0=0.1)

	som.view_umatrix(bestmatches=False,labels=numeric_labels,colorbar=True)

def ex2_4():
	"""variação do decaimento da taxa de aprendizagem"""

	n_rows, n_columns = 20, 20

	filename = "isolet1_2_3_4.data"

	names=[]
	for i in range(1,618):
	    names.append("f"+str(i))

	names.append("letter_class")

	dataset = pd.read_csv(filename, header=None, names=names)
	dataset.head()

	labels=dataset.letter_class.values
	colormap=cm.ScalarMappable(cmap="tab20b")


	# seaborn.pairplot(dataset,hue="species", height=2, diag_kind="kde")
	# plt.show()
	from sklearn.preprocessing import normalize

	type_labels, numeric_labels=np.unique(labels, return_inverse=True)
	# print(numeric_labels)
	str_labels= [ "c"+str(e) for e in numeric_labels]

	print("type_labels",type_labels)
	labels_legend=[]
	for i,label in enumerate(type_labels):
		labels_legend.append(tuple([str(i),str(int(label))]))
	print(labels_legend)


	FEATURES = dataset.columns[0:617]
	x_data = dataset[FEATURES].as_matrix()
	x_data = normalize(x_data)


	som = somoclu.Somoclu(n_columns, n_rows, compactsupport=False,neighborhood="gaussian")
	som.train(x_data,radiuscooling="linear",scalecooling="exponential",scale0=0.1)

	som.view_umatrix(bestmatches=False,labels=numeric_labels,colorbar=True)

def ex2_5():
	"""variação da taxa de aprendizagem:0.2"""

	n_rows, n_columns = 20, 20

	filename = "isolet1_2_3_4.data"

	names=[]
	for i in range(1,618):
	    names.append("f"+str(i))

	names.append("letter_class")

	dataset = pd.read_csv(filename, header=None, names=names)
	dataset.head()

	labels=dataset.letter_class.values
	colormap=cm.ScalarMappable(cmap="tab20b")

	labels=list(map(int,labels))


	# seaborn.pairplot(dataset,hue="species", height=2, diag_kind="kde")
	# plt.show()
	from sklearn.preprocessing import normalize

	type_labels, numeric_labels=np.unique(labels, return_inverse=True)
	# print(numeric_labels)
	str_labels= [ "c"+str(e) for e in numeric_labels]

	print("type_labels",type_labels)
	labels_legend=[]
	for i,label in enumerate(type_labels):
		labels_legend.append(tuple([str(i),str(int(label))]))
	print(labels_legend)


	FEATURES = dataset.columns[0:617]
	x_data = dataset[FEATURES].as_matrix()
	x_data = normalize(x_data)


	som = somoclu.Somoclu(n_columns, n_rows, compactsupport=False,neighborhood="gaussian")
	som.train(x_data,radiuscooling="linear",scalecooling="linear",scale0=0.2)

	som.view_umatrix(bestmatches=False,labels=labels,colorbar=True)

def ex3_1():

	n_rows, n_columns = 20, 20



	filename = "segmentation.data"
	names=["segment"]
	names.extend(["region-centroid-col","region-centroid-row","region-pixel-count",
	"short-line-density-5","short-line-density-2","vedge-mean","vegde-sd",
	"hedge-mean","hedge-sd","intensity-mean","rawred-mean","rawblue-mean",
	"rawgreen-mean","exred-mean","exblue-mean","exgreen-mean","value-mean",
	"saturatoin-mean","hue-mean"])

	dataset = pd.read_csv(filename, header=None, names=names)
	dataset.head()




	labels=dataset.segment.values
	colormap=cm.ScalarMappable(cmap="tab20b")




	# seaborn.pairplot(dataset,hue="species", height=2, diag_kind="kde")
	# plt.show()
	from sklearn.preprocessing import normalize

	type_labels, numeric_labels=np.unique(labels, return_inverse=True)
	# print(numeric_labels)
	print(type_labels)
	from collections import OrderedDict 
	labels_legend=[]
	for i,label in enumerate(type_labels):
		labels_legend.append(tuple([str(i),label]))
	print(labels_legend)

	str_labels= [ "c"+str(e) for e in numeric_labels]


	FEATURES = dataset.columns[1:20]
	x_data = dataset[FEATURES].as_matrix()
	x_data = normalize(x_data)


	som = somoclu.Somoclu(n_columns, n_rows, compactsupport=False)
	som.train(x_data)


	som.view_umatrix(bestmatches=False,labels=numeric_labels, colorbar=True,labels_legend=labels_legend)

def ex3_2():
	"""variação do decaimento da função de vizinhança: exponencial"""

	n_rows, n_columns = 20, 20

	filename = "segmentation.data"
	names=["segment"]
	names.extend(["region-centroid-col","region-centroid-row","region-pixel-count",
	"short-line-density-5","short-line-density-2","vedge-mean","vegde-sd",
	"hedge-mean","hedge-sd","intensity-mean","rawred-mean","rawblue-mean",
	"rawgreen-mean","exred-mean","exblue-mean","exgreen-mean","value-mean",
	"saturatoin-mean","hue-mean"])

	dataset = pd.read_csv(filename, header=None, names=names)
	dataset.head()




	labels=dataset.segment.values

	colormap=cm.ScalarMappable(cmap="tab20b")




	# seaborn.pairplot(dataset,hue="species", height=2, diag_kind="kde")
	# plt.show()
	from sklearn.preprocessing import normalize

	type_labels, numeric_labels=np.unique(labels, return_inverse=True)
	# print(numeric_labels)
	print(type_labels)
	from collections import OrderedDict 
	labels_legend=[]
	for i,label in enumerate(type_labels):
		labels_legend.append(tuple([str(i),label]))
	print(labels_legend)

	str_labels= [ "c"+str(e) for e in numeric_labels]


	FEATURES = dataset.columns[1:20]
	x_data = dataset[FEATURES].as_matrix()
	x_data = normalize(x_data)


	som = somoclu.Somoclu(n_columns, n_rows, compactsupport=False,neighborhood="gaussian")
	som.train(x_data,radiuscooling="exponential")

	som.view_umatrix(bestmatches=False,labels=numeric_labels,colorbar=True,
		labels_legend=labels_legend)


def ex3_3():
	"""variação da função de vizinhança: bubble"""

	n_rows, n_columns = 20, 20

	filename = "segmentation.data"
	names=["segment"]
	names.extend(["region-centroid-col","region-centroid-row","region-pixel-count",
	"short-line-density-5","short-line-density-2","vedge-mean","vegde-sd",
	"hedge-mean","hedge-sd","intensity-mean","rawred-mean","rawblue-mean",
	"rawgreen-mean","exred-mean","exblue-mean","exgreen-mean","value-mean",
	"saturatoin-mean","hue-mean"])

	dataset = pd.read_csv(filename, header=None, names=names)
	dataset.head()




	labels=dataset.segment.values

	colormap=cm.ScalarMappable(cmap="tab20b")




	# seaborn.pairplot(dataset,hue="species", height=2, diag_kind="kde")
	# plt.show()
	from sklearn.preprocessing import normalize

	type_labels, numeric_labels=np.unique(labels, return_inverse=True)
	# print(numeric_labels)
	print(type_labels)
	from collections import OrderedDict 
	labels_legend=[]
	for i,label in enumerate(type_labels):
		labels_legend.append(tuple([str(i),label]))
	print(labels_legend)

	str_labels= [ "c"+str(e) for e in numeric_labels]


	FEATURES = dataset.columns[1:20]
	x_data = dataset[FEATURES].as_matrix()
	x_data = normalize(x_data)



	som = somoclu.Somoclu(n_columns, n_rows, compactsupport=False,neighborhood="bubble")
	som.train(x_data,radiuscooling="linear",scalecooling="linear",scale0=0.1)

	som.view_umatrix(bestmatches=False,labels=numeric_labels,colorbar=True,
		labels_legend=labels_legend)

def ex3_4():
	"""variação do decaimento da taxa de aprendizagem"""

	n_rows, n_columns = 20, 20

	filename = "segmentation.data"
	names=["segment"]
	names.extend(["region-centroid-col","region-centroid-row","region-pixel-count",
	"short-line-density-5","short-line-density-2","vedge-mean","vegde-sd",
	"hedge-mean","hedge-sd","intensity-mean","rawred-mean","rawblue-mean",
	"rawgreen-mean","exred-mean","exblue-mean","exgreen-mean","value-mean",
	"saturatoin-mean","hue-mean"])

	dataset = pd.read_csv(filename, header=None, names=names)
	dataset.head()




	labels=dataset.segment.values

	colormap=cm.ScalarMappable(cmap="tab20b")




	# seaborn.pairplot(dataset,hue="species", height=2, diag_kind="kde")
	# plt.show()
	from sklearn.preprocessing import normalize

	type_labels, numeric_labels=np.unique(labels, return_inverse=True)
	# print(numeric_labels)
	print(type_labels)
	from collections import OrderedDict 
	labels_legend=[]
	for i,label in enumerate(type_labels):
		labels_legend.append(tuple([str(i),label]))
	print(labels_legend)

	str_labels= [ "c"+str(e) for e in numeric_labels]


	FEATURES = dataset.columns[1:20]
	x_data = dataset[FEATURES].as_matrix()
	x_data = normalize(x_data)



	som = somoclu.Somoclu(n_columns, n_rows, compactsupport=False,neighborhood="gaussian")
	som.train(x_data,radiuscooling="linear",scalecooling="exponential",scale0=0.1)

	som.view_umatrix(bestmatches=False,labels=numeric_labels,colorbar=True,
		labels_legend=labels_legend)

def ex3_5():
	"""variação da taxa de aprendizagem:0.2"""

	n_rows, n_columns = 20, 20

	filename = "segmentation.data"
	names=["segment"]
	names.extend(["region-centroid-col","region-centroid-row","region-pixel-count",
	"short-line-density-5","short-line-density-2","vedge-mean","vegde-sd",
	"hedge-mean","hedge-sd","intensity-mean","rawred-mean","rawblue-mean",
	"rawgreen-mean","exred-mean","exblue-mean","exgreen-mean","value-mean",
	"saturatoin-mean","hue-mean"])

	dataset = pd.read_csv(filename, header=None, names=names)
	dataset.head()




	labels=dataset.segment.values

	colormap=cm.ScalarMappable(cmap="tab20b")




	# seaborn.pairplot(dataset,hue="species", height=2, diag_kind="kde")
	# plt.show()
	from sklearn.preprocessing import normalize

	type_labels, numeric_labels=np.unique(labels, return_inverse=True)
	# print(numeric_labels)
	print(type_labels)
	from collections import OrderedDict 
	labels_legend=[]
	for i,label in enumerate(type_labels):
		labels_legend.append(tuple([str(i),label]))
	print(labels_legend)

	str_labels= [ "c"+str(e) for e in numeric_labels]


	FEATURES = dataset.columns[1:20]
	x_data = dataset[FEATURES].as_matrix()
	x_data = normalize(x_data)



	som = somoclu.Somoclu(n_columns, n_rows, compactsupport=False,neighborhood="gaussian")
	som.train(x_data,radiuscooling="linear",scalecooling="linear",scale0=0.2)

	som.view_umatrix(bestmatches=False,labels=numeric_labels,colorbar=True,
		labels_legend=labels_legend)



if __name__=="__main__":
	# ex1_1()
	# ex2_1()
	ex3_1()


