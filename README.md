# Soil Analysis
The soil data is stored as a csv file and was analyzed using various Python modules.
The Pandas Python module vX.X was used to process and handle the raw data read in from the csv file.
Matplotlib vX.X along with the Seaborn package vX.X was used to visualize the data.
Scikit Learn was leveraged to perform Principal Component Analysis and KMeans clustering of the data.

The modules were installed using pip according to there documentation and imported into the `analysis.py` script as such

```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA 
```

The analysis can be reproduced by downloading the soil dataset csv file and running `python analysis.py`.

### Descriptive Statistics
The soil dataset consists of 21 columns of data.
Each separate region in the park is denoted as a shorthand (cw) (tb) etc...
Columns 0-3 tag each sample with information about where it was extracted.
The transect and plot give the detailed location of each sample which are uniquely identifiable by there sample id.


The provided csv file was read into a Pandas dataframe using 

`df = pd.read_csv('vcp_soil_ppm.csv')`

Columns 4-20 in Figure x hold the measurements in parts per million (ppm) for the concentration of each element sampled.
This summary of the dataset can be created using the method `df.info()` on the Pandas dataframe.

```
[180 rows x 21 columns]
General Dataset Description
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 180 entries, 0 to 179
Data columns (total 21 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   region    180 non-null    object 
 1   transect  180 non-null    object 
 2   plot      180 non-null    object 
 3   sample    180 non-null    object 
 4   Na        180 non-null    float64
 5   Mg        180 non-null    float64
 6   Al        180 non-null    float64
 7   Si        180 non-null    float64
 8   S         180 non-null    float64
 9   K         180 non-null    float64
 10  Ca        180 non-null    float64
 11  Ba        180 non-null    float64
 12  Ti        180 non-null    float64
 13  V         180 non-null    float64
 14  Cr        180 non-null    float64
 15  Mn        180 non-null    float64
 16  Fe        180 non-null    float64
 17  Co        180 non-null    float64
 18  Ni        180 non-null    float64
 19  Cu        180 non-null    float64
 20  Zn        180 non-null    float64
```

There are 180 non null entries in total which match the number of samples collected and measured.
This dataset therefore can be said to have 180 datapoints each with 17 numerical features and 4 descriptive labels.
In this study each of the 4 regions will be considered independantly.

The highest average concentration of elements accross all sites was of silicon (162,051 ppm) and iron (31,078 ppm).
Aluminum (3 ppm) and cobalt (25 ppm) were the elements with the lowest average concentration accross all regions.

The complete breakdown of statistics for each element in the dataset can be found in Figure x.

```
                Na            Mg          Al             Si            S  \
count   180.000000    180.000000  180.000000     180.000000   180.000000   
mean   1101.007274  12132.487882    3.068844  162051.079607  7855.029377   
std     724.544757   2318.695823    0.482190   29729.546405   465.544185   
min    -508.411700   8099.008160    1.435630   78952.344680  7194.559470   
25%     584.663127  10373.846767    2.769082  146722.686125  7511.008855   
50%    1012.834400  11857.072045    3.077606  167518.089400  7737.765955   
75%    1618.803772  13614.821383    3.361886  181297.825225  8061.059775   
max    2818.509530  18088.716750    4.957717  215358.753900  9369.779190   

                  K            Ca           Ba           Ti           V  \
count    180.000000    180.000000   180.000000   180.000000  180.000000   
mean   10813.282653   8002.330875  1189.625992  4719.168235  162.387222   
std     2118.892400   7813.202305   400.386059   953.951702   68.049288   
min     5463.809610    976.421910   418.973124  2660.864710  -13.453805   
25%     9311.399040   3916.672990   947.215025  4192.900558  116.589469   
50%    10478.258590   6155.832385  1105.607579  4534.223505  155.628452   
75%    12266.201937   8478.934090  1300.355000  5135.185385  198.075475   
max    16696.012740  45469.269500  3387.880962  9869.866740  430.856009   

               Cr            Mn            Fe          Co          Ni  \
count  180.000000    180.000000    180.000000  180.000000  180.000000   
mean    63.482892   1326.906435  31078.571850   25.483146   49.494488   
std     16.274243   1632.883224   7491.401488    5.036103   15.179858   
min    -12.245582     49.986346  15937.687030   12.707727   -1.798847   
25%     55.469601    547.109999  25903.509178   22.573605   38.982259   
50%     65.248698    967.319192  30250.111995   25.030128   48.335646   
75%     73.044311   1505.364815  34423.850740   28.183969   59.079205   
max     96.480604  12727.966900  54920.520290   39.431235   90.693063   

               Cu           Zn  
count  180.000000   180.000000  
mean   180.255026   299.005147  
std    144.472118   153.855002  
min    -10.125255   143.451616  
25%    115.967538   236.984295  
50%    143.255809   254.744146  
75%    182.190312   295.190591  
max    974.460895  1211.514519  

```
This table of descriptive statistics was generated using `df.describe()`.

Why do some have the lowest value in the negatives??
Histograms of the raw data can be found in Appendix.

![All Elements by Region](results/dataset-visual.png?raw=true)

TODO: Add comment here as to how each region is different by element or the same by comparing each elements regional mean to the overall mean.

### Element Correlation

The Pearson correlation coefficient can be used to describe the linear relationship between two independant variables.
It value ranges from -1 to 1 with -1 being total negative linear correlation and +1 being total positive linear correlation.
Positive linear correlation describes the relationship between two variables in which if the value of one increases, so does the other.
The opposite is also said to be true in that a descread in one positively correlated linear variable results in a descrease of the second as well.

The results of calculating this coefficient for each combinational pair of elements results in the following heatmap.
This correlation matrix was calculated using Pandas and plotted using the Seaborn package.

```
# Determine the correlation of the dataframe
correlation = df.corr()

# Plot the correlation heatmap
sns.heatmap(round(correlation, 2), annot=True, cmap="coolwarm", fmt=".2f", linewidth=.03)
plt.show()
```

Inspecting the correlation heatmap for the entire dataset in Figure X shows that there is only one element pair that is strongly correlated.

![All Region Element Heatmap](results/all-regions-correlation.png?raw=true)

Strong correlation refers to correlations with magnitudes greater than 0.90.
Sodium (Na) and Magnesium (Mg) show a strong negative linear correlation for the complete dataset.
This relationship can be shown to be strong accross each region as well when inspected independantly of one another as shown in Figures A, B, C, and D.
No other element shows strong correlation, although the Copper (Cu) - Zinc (Zn) and Cadanium (Ca) - Zinc (Zn) have high correlations at 0.89.

The Cu-Zn pair is only strongly positively correlated in one region,  nwf.
Each of the other three regions show weak or mild correlation; vh (0.56), cw (0.27), and tb (0.10).
It can therefore be inferred by visual inspection of the correlation heatmaps that highly correlated pairs of elements can potentially be used for distictly classifiying each region.
Each region has its own unique pair of highly correlated elements as shown in Figures, A, B, C, and D.


![NWF Region Element Heatmap](./results/nwf-correlation.png?raw=true)
![VH Region Element Heatmap](./results/vh-correlation.png?raw=true)
![TB Region Element Heatmap](./results/tb-correlation.png?raw=true)
![CW Region Element Heatmap](./results/cw-correlation.png?raw=true)


As discussed the nwf region has a high positive linear correlation amongst the Cu-Zn element pair in which no other region has a signigicantly strong correlation.
The CW region has a uniquely high correlation between Titanium (Ti) and Nickel (Ni) at 0.93.
Both the VH and TB regions have strong positive correlations to iron (Fe) and Cobalt (Co) although they do also have uniquely strong correlated pairs in Mn-Co at 0.91 for TB and AL-K at 0.92 for VH.

It would seem from even this preliminary analysis that uniquely strong positive correlation pairs between each region can be useful for classification of the respective regions.


### Principal Component Analysis
Although there are many element pairs which show only slight to moderate correlation, it may still be useful to reduce the amount of redudant information stored in each variable by analyzing 
the amount of variance each element explains in the dataset.

Principal Component Analysis (PCA) is a useful data reduction tool that creates orthogonal linear combinations of features to create new dimensions for the underlying dataset.
This can be used to remove elements which are highly correlated and therefore provide redundant information to the dataset.
This data reduction can be useful for visualizating underlying dataset relationships and provide performance improvements during future processing steps such as KMeans clustering.
PCA can also provide insight into which feature groups are used to make up each new dimension.

PCA uses eigen values and vectors etc....

The explained variance of each principal component that is created can be used to show the amount of dimensionality reduction that can be performed on the initial feature set.
The overall goal is to use the minimum amount of pricipal components while maintaining a 90% explained variance on the data.
Calulcating the Principal Components can be accomplished using the Python SciKit Learn Library 

```
pca = PCA() 
components = pca.fit(X)
```

where `X` is a subset of the original dataframe that includes all the features, but none of the labels.
The features are scaled using the `StandardScaler` method of the SciKit Learn preprocessing library which removes the mean and scales the features according to the unit variance.

`X = preprocessing.StandardScaler().fit_transform(df.loc[:,df.dtypes == 'float64'])`

Once the dataset has been split into orthogonal components using PCA, the resulting variance of the new components can be shown on a histogram

```
n = [x for x in range(1, pca.n_components_+1)]
plt.bar(n, pca.explained_variance_ratio_, color="black")
plt.xticks(n)
plt.show()
```

The minimal amount of Python code needed for graphing the results of PCA are shown here. Full code can be found in the Appendix.

![All Data Feature Correlation Heatmap](results/all-regions-correlation.png?raw=true)

```
Explained Variance Percentage of Each Principal Component

PC-1: 32.23%
PC-2: 26.23%
PC-3: 12.73%
PC-4: 10.69%
PC-5: 7.34%
PC-6: 2.95%
PC-7: 2.25%
PC-8: 1.58%
PC-9: 0.95%
PC-10: 0.82%
PC-11: 0.67%
PC-12: 0.52%
PC-13: 0.39%
PC-14: 0.34%
PC-15: 0.19%
PC-16: 0.08%
PC-17: 0.05%
```

It is shown in Figure X. that in order to preserve around 90% of the explained variance of the original dataset, the first five principal components must be used.
The first five components account for 89.22% of the total explained variance.
This represents a reduction in dimensionality of 12 dimensions of the data from the original 17 features down to just 5 features.

Factor loadings describe the weighting on each linear combination of elements that is created from performing PCA.
Each component represents a linear combination of orthogonal features from the original dataset.
These factor weightings are shown in Figure X.

![PCA Component Weighting](results/pca-heatmap-components.png)

As the Na-Mg correlation showed a highly negative linear relationship in the previous correlation heatmaps for each region, it provides little information for distinguishing between 
each region.
The overall total concentration for each region also varys only slightly.
It is therefore expected that this combination of does not provide much insight into the overall dataset.
This is verified in Figure X. as both Na and Mg are the most highly weighted elements in the 16th principal component which explains only 0.08% of the variance.
These elements are the most significantly weighted in the 16th principal component.

Observing the highest weights (>0.35) on the first 5 factor loadings shows each component is made up of unique elements.
Looking at the first five components shows that the first principal component is mainly made up of S (0.37) and Cu (0.37). (non-metal and transition metal)
The second component is made up of Al (-0.36) and Ni (-0.38). (post transition metal and transition metal)
This vector potentially represents the null vector with some noise. 
The third component is made up of mostly Cr (-0.38), Mn (0.47), and Co (0.42). (transition elements 24,25,27)
The fourth component is made up of mostly Mn (0.45), Ca (0.40), and Zn (0.36). (transition metal, Akaline earth metal, transition metal)
The fifth component is made up of mostly Al (0.42), Ti (0.38), Fe (-0.49), and Co (-0.42) (post transition metal, transition metal, transition metal, transition metal)


Interpreting the weights assigned to each principal component requires domain expertise and is sometimes not possible.
All components do have a transition metal as one of the most heavily weighted components along with a unique element class.
Weights can be both positive and negative to show positive and negative underlying correlation.
The first component can be considered the non-metal weighting, the second the negative post-transition metal weighting, the third comprising middle transition elements,
the fourth a combination of akaline earth metal and transition metal, and the fifth the positive post transition metals.

This represents a simplified interpretation of PCA components.

### K-Means Clustering

K-Means is an exact solution to linear problem when looked at in relation to PCA....etc

K Means clustering...


![All Data KMeans Inertia](results/kmeans-all-data-inertia.png)
Homogeneity Score: 0.265
Completeness Score: 0.290
V Score: 0.277

##### PCA KMeans
![PCA KMeans Inertia](results/pca-kmeans-inertia.png)
![PCA KMeans Clustering](results/pca-kmeans-scatter-plot.png)

PCA KMeans Performance
Homogeneity Score: 0.258 \
Completeness Score: 0.304 \
V Score: 0.279


