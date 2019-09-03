import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, learning_curve
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import statsmodels.api as sm


#### KMEANS Imports ######
from sklearn.cluster import KMeans

#### SVC imports #######
from sklearn import svm

df = pd.read_csv('soil_vcp_allsites_2018_loc.csv')
# Remove unnamed values
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.drop(columns=['transect', 'plot', 'sample'])

#print(f'df: {df}')
y= df.loc[:, df.columns == 'region']
#print(f'y: {y}')

X = df.loc[:, df.columns != 'region']
#print(f'X: {X}')

#print(X)
##### Univariate ANalysis #########
"""
X.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0,
                       xlabelsize=8, ylabelsize=8, grid=False)    
plt.tight_layout() 
plt.show()
"""
##################################
# Normalize the data
from sklearn import preprocessing
'''
for region in ['cw', 'nwf', 'vh', 'tb']:
    #tmp = df.loc[df['region'] == region]
    tmp = df[df['region'] == region]
    tmp = tmp.loc[:, tmp.columns != 'region'] 
    print(f'TMP: {tmp}')
    data = pd.DataFrame(preprocessing.scale(tmp),columns = tmp.columns) 
    
    ##### Univariate ANalysis #########
    
    data.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0,
                           xlabelsize=8, ylabelsize=8, grid=False)    
    plt.suptitle(f'Normalized Histograms of Samples - {region}')
    plt.xlabel('Measurement')
    plt.ylabel('n samples')
    plt.tight_layout()
    #plt.show()
    ##################################
    ########### Correlation Matrix OPtion 1 ###################
    """
    print(data)
    """
    corr = data.corr()
    """
    print(corr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,len(data.columns),1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(data.columns)
    ax.set_yticklabels(data.columns)
    plt.show()
    """
    #############################################################
    ############ Correlation Matrix Option 2 #####################
    f, ax = plt.subplots(figsize=(10, 6))
    hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                                      linewidths=.03)
    f.subplots_adjust(top=0.93)
    t= f.suptitle(f'Soil Composition Correlation - {region}', fontsize=14)
    #plt.show()
'''

pca = PCA(n_components=2)
#tmp = X.loc[:, df.columns != 'region']
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:,0], X_pca[:,1])
plt.show()


pcsummary = pd.DataFrame(pca.components_,columns=X_pca.columns,index = ['PC-1','PC-2'])
print("PCA", pca.explained_variance_ratio_)
f = open('pca.tex', 'w')
f.write(pcsummary.to_latex())
f.close()



#####  Now Let's create labels for kmeans clustering #####
y.loc[df['region'] == 'tb', 'region'] = 0
y.loc[df['region'] == 'vh', 'region'] = 1
y.loc[df['region'] == 'nwf', 'region'] = 2
y.loc[df['region'] == 'cw', 'region'] = 3
print(y)

# ^^^ Might not need to do that #
print(X_pca)
kmeans = KMeans(n_clusters=4).fit(X_pca) 
print(kmeans.labels_)
print(kmeans.cluster_centers_)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:,0], centroids[:,1], zorder=10, marker='x', c='r', linewidths=2)
#plt.scatter(X[:,0],X[:,1], c=kmeans.labels_)
plt.scatter(X_pca[:,0],X_pca[:,1])
plt.show()

##### But its better to use SVC to gauge accuracy #########
'''
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.4, random_state=0)
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
score = clf.score(X_test, y_test)
print("SVC Classifier results")
print(f"SVC SCORE: {score}")

### Cross validate
scores = cross_val_score(clf, X, y.values.ravel(), cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
'''
