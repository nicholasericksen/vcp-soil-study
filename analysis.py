import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import itertools
import random

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import v_measure_score


label_key = input("Enter label key ('region' or 'soil_type'): ")
LABELS_KEY = label_key
DROP_COLS = ["region", "soil_type", "site", "transect", "plot", "sample"]
DROP_COLS.remove(label_key)
#TODO add logic to grab list of unique labels and plot each heatmap and label separetely.
# Read data from soil csv into dataframe and print it
print("Reading csv...")
df = pd.read_csv('data/vcp_soil_ppm_with_soil_type.csv')
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
plt.title(f"Elemental Concentration Per {LABELS_KEY}")
plt.xlabel("Element")
plt.ylabel("Concentration in Parts Per Million (PPM)")
pd.plotting.parallel_coordinates(df.loc[:, ~df.columns.isin(DROP_COLS)], LABELS_KEY, color=('#556270', '#4ECDC4', '#C7F464', '#ECB445', '#a96836', '#7bcc71'))
#pd.plotting.parallel_coordinates(df.loc[:,df.dtypes == 'float64'], LABELS_KEY, color=('#556270', '#4ECDC4', '#C7F464', '#ECB445'), alpha=0.5)
plt.savefig(f"results/dataset-visual-{LABELS_KEY}.png")
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
cluster_amount = int(input("Enter number of clusters to test: "))
#cluster_amount = 7
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
print(f"KMeans Raw Data Performance ({LABELS_KEY})")
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

pca_components = int(input("Enter the number of pca_components to include: ")) - 1


for k in ks:
  model = KMeans(n_clusters=k)
  model.fit(X_pca[:,0:pca_components])
  inertias.append(model.inertia_)

plt.plot(ks, inertias, '-o', color='black')
plt.title("Inertia for 'n' PCA Clusters")
plt.xlabel('Number of Clusters, k')
plt.ylabel('Inertia')
plt.xticks(ks)
plt.savefig("results/pca-kmeans-inertia.png")
plt.show()

#cluster_amount = int(input("What is the ideal amount of clusters?: "))
#cluster_amount = 7
# For now use same cluster amount as raw data
#TODO make this 7 clusters to match curve elbow and number of different soil types
model = KMeans(n_clusters=cluster_amount)
#TODO check this slicing grabs first 5 pca components
clusters = model.fit(X_pca[:,0:pca_components])
y_pred = clusters.labels_

# Measuring KMeans Performance
print("\nPCA KMeans Performance")
print("Homogeneity Score")
print(homogeneity_score(y_true, y_pred))
print("Completeness Score")
print(completeness_score(y_true, y_pred))
print("V Score")
print(v_measure_score(y_true, y_pred))

# Plot Kmeans scatterplot for first 2 PCA
fig, ax = plt.subplots()

index = 0

current_label = y[LABELS_KEY][0]
X_pca_1_tmp = []
X_pca_2_tmp = []

marker = itertools.cycle(('+', 'o', '*', 'v', '<', '>', 'x', 'D')) 

r = lambda: random.randint(0,255)
colors = ['#%02X%02X%02X' % (r(),r(),r()) for x in range(0, len(unique_labels))]

#TODO perform pca on entire df instead of joining parameters
df_pca = pd.DataFrame(X_pca)
df_pca['y_pred'] = y_pred 
df_pca['label'] = y[LABELS_KEY]
df_pca['y_true'] = y_true
df_pca['color'] = y_pred

df_pca['color'] = df_pca['color'].replace(numerical_labels, [colors[x] for x in range(0, len(unique_labels))])
print(unique_labels)
print(df_pca)

handles = []
for label in unique_labels:
  tmp_df = df_pca[df_pca["label"] == label]
  m = next(marker)
  scatter = ax.scatter(tmp_df[0], tmp_df[1], label=label, marker=m, color=tmp_df["color"])
  handles += ([mlines.Line2D([],[],color="black", marker=m, label=label, linestyle='')])
  print(tmp_df.columns)

plt.legend(handles=handles, title="Classes")

centroids = clusters.cluster_centers_
plt.scatter(centroids[:,0], centroids[:,1], zorder=10, marker='x', c='r', linewidths=2)
#TODO: plot also the kmeans fit data on as different symbol maybe?
plt.xlabel("PC-1")
plt.ylabel("PC-2")
plt.title("First 2 Principal Components Clustered with KMeans")
plt.savefig(f"results/pca-kmeans-scatter-plot-{LABELS_KEY}.png")
plt.annotate(f"colors represent n_clusters (n={cluster_amount})", xy=(0.65, 0.03), xycoords='axes fraction')
#plt.text(0.02, 0.5, "Test string testing", fontsize=14, transform=plt.gcf().transFigure)
plt.show()

