import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.decomposition import PCA

#df = pd.read_csv('soil_vcp_allsites_2018_loc.csv')
df = pd.read_csv('data.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.drop(columns=['transect', 'plot', 'sample'])
df.loc[:, df.columns != 'region'] = preprocessing.scale(df.loc[:, df.columns != 'region'])

pca = PCA(n_components=2)
pca.fit(df.loc[:, df.columns != 'region'])

print(f'Explained vaiance \n: {pca.explained_variance_ratio_}')
print(f'Singular Valus \n: {pca.singular_values_}')

columns = ['pca_%i' % i for i in range(2)]
df_pca = pd.DataFrame(pca.transform(df.loc[:, df.columns != 'region']), columns=columns, index=df.index)

#df_pca.insert(df['region'])
df_pca['region'] = df['region']

for region in df_pca.region.unique():
    data = df_pca.loc[df_pca['region'] == region]
    plt.scatter(data['pca_0'], data['pca_1'], label=region)
#    plt.scatter(new.loc[new['region'] == region], label=region)

print(pca.components_)
components = pd.DataFrame(pca.components_,columns=df.columns[1:], index = ['PC-1','PC-2'])
print(components)
#print(np.abs(components['PC-1']))

plt.legend()
plt.title('Principle Component Analysis')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()


