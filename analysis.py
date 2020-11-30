import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

LABELS_KEY = "region"
DROP_COLS = ["sample", "transect", "plot", "sample"]
#TODO add logic to grab list of unique labels and plot each heatmap and label separetely.
# Read data from soil csv into dataframe and print it
print("Reading csv...")
df = pd.read_csv('data/vcp_soil_ppm.csv')
print(df)


# Set pandas to show all data columns
column_length = (len(df.columns))
pd.set_option('display.max_columns', column_length)

# Show the general info of the dataset
print("General Dataset Description")
print(df.info())

# Show the descriptive statistics for the dataset
print(df.describe())

# Plot all features using Parallel Coordinates
f, ax = plt.subplots(figsize=(10,6))
plt.title("Elemental Concentration Per Region")
plt.xlabel("Element")
plt.ylabel("Concentration in Parts Per Million (PPM)")
pd.plotting.parallel_coordinates(df.loc[:, ~df.columns.isin(DROP_COLS)], LABELS_KEY, color=('#556270', '#4ECDC4', '#C7F464', '#ECB445'), alpha=0.5)
plt.savefig("results/dataset-visual.png")
plt.show()

# Correlation heatmap for entire dataset
correlation = df.corr()
f, ax = plt.subplots(figsize=(10,6))
sns.heatmap(round(correlation, 2), annot=True, cmap="coolwarm", fmt=".2f", linewidth=.03)
plt.title("Pearsons Correlation Coefficient Feature Heatmap")
plt.savefig("results/all-regions-correlation.png")
plt.show()

# Preprocessing dataset
X = preprocessing.StandardScaler().fit_transform(df.loc[:,df.dtypes == 'float64'])
y = df.loc[:, df.columns == LABELS_KEY]

# PCA of dataset
pca = PCA()

components = pca.fit(X)

# Plot explained variance of each component
#n = range(pca.n_components_)
f, ax = plt.subplots(figsize=(10,6))
n = [x for x in range(1, pca.n_components_+1)]
plt.bar(n, pca.explained_variance_ratio_, color="black")
plt.title("Principal Components")
plt.xlabel("PCA Features")
plt.ylabel("variance %")
plt.xticks(n)
plt.savefig("results/pca-explained-variance.png")
plt.show()

print("\nExplained Variance Percentage of Each Principal Component\n")
for index, var in enumerate(pca.explained_variance_ratio_):
  print(f"PC-{index + 1}: {round(var * 100, 2)}%")

  


# Plot element weights
f, ax = plt.subplots(figsize=(10,6))
sns.heatmap(pca.components_,
                 cmap='YlGnBu',
                 yticklabels=["PCA-"+str(x) for x in n],
                 xticklabels=list(df.loc[:,df.dtypes == 'float64']),
                 fmt=".2f",
                 annot=True,
                 linewidth=.03)
plt.title("Element Combinations of Each Principal Component")
#ax.set_aspect("equal")
plt.savefig("results/pca-heatmap-components.png")
plt.show()

# Kmeans of original dataset

# Find optimal number of clusters using inertia
ks = range(1,20)
inertias = []

for k in ks:
  model = KMeans(n_clusters=k)
  model.fit(X)
  inertias.append(model.inertia_)

plt.plot(ks, inertias, '-o', color='black')
plt.title("Inertia for 'n' Clusters")
plt.xlabel('Number of Clusters, k')
plt.ylabel('Inertia')
plt.xticks(ks)
plt.savefig("results/kmeans-all-data-inertia.png")
plt.show()

# Plot Kmeans scatterplot
#cluster_amount = int(input("What is the ideal amount of clusters?: "))
cluster_amount = 4
#TODO make this 7 clusters to match curve elbow and number of different soil types
model = KMeans(n_clusters=cluster_amount)
clusters = model.fit(X)
y_pred = clusters.labels_

# Creating numerical labels for original data
y_true = []

unique_labels = list(y[LABELS_KEY].unique())
numerical_labels = [i for i in range(0, len(unique_labels))]

for i, label in enumerate(unique_labels):
  print(f"{label}: {numerical_labels[i]}")

for label in y[LABELS_KEY]:
  y_true.append(numerical_labels[unique_labels.index(label)])

# Measuring KMeans Performance
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import v_measure_score

print("Homogeneity Score")
print(homogeneity_score(y_true, y_pred))
print("Completeness Score")
print(completeness_score(y_true, y_pred))
print("V Score")
print(v_measure_score(y_true, y_pred))











#TODO refactor this into one function above
# Kmeans of PCA dataset
X_pca = pca.transform(X)

ks = range(1,20)
inertias = []

for k in ks:
  model = KMeans(n_clusters=k)
  model.fit(X_pca[:,0:4])
  inertias.append(model.inertia_)

plt.plot(ks, inertias, '-o', color='black')
plt.title("Inertia for 'n' PCA Clusters")
plt.xlabel('Number of Clusters, k')
plt.ylabel('Inertia')
plt.xticks(ks)
plt.savefig("results/pca-kmeans-inertia.png")
plt.show()

#cluster_amount = int(input("What is the ideal amount of clusters?: "))
cluster_amount = 4
#TODO make this 7 clusters to match curve elbow and number of different soil types
model = KMeans(n_clusters=cluster_amount)
#TODO check this slicing grabs first 5 pca components
clusters = model.fit(X[:])
y_pred = clusters.labels_

# Measuring KMeans Performance
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import v_measure_score

print("\nPCA KMeans Performance")
print("Homogeneity Score")
print(homogeneity_score(y_true, y_pred))
print("Completeness Score")
print(completeness_score(y_true, y_pred))
print("V Score")
print(v_measure_score(y_true, y_pred))

# Plot Kmeans scatterplot for first 2 PCA



fig, ax = plt.subplots()

scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=y_true, label=y[LABELS_KEY])

handles, labels = scatter.legend_elements()
legend1 = ax.legend(handles, unique_labels,loc="lower left", title="Classes")
ax.add_artist(legend1)
centroids = clusters.cluster_centers_
plt.scatter(centroids[:,0], centroids[:,1], zorder=10, marker='x', c='r', linewidths=2)
#TODO: plot also the kmeans fit data on as different symbol maybe?
plt.xlabel("PC-1")
plt.ylabel("PC-2")
plt.title("First 2 Principal Components Clustered with KMeans")
plt.savefig("results/pca-kmeans-scatter-plot.png")
plt.show()

