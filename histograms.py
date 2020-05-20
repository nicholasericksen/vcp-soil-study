import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set()
pd.set_option('display.max_columns', 25)

data = pd.read_csv('vcp_soil_ppm.csv')

bg = pd.read_csv('vcp_bedrock_ppm.csv')

# Convert this whole idea into just a conditional clause such as
# " where mean > 1000 " to eliminate trace elements
labels = ['region', 'Mn', 'Fe', 'Ti', 'Ba', 'Ca', 'K', 'S', 'Si', 'Mg']
# can then eliminate copying the dataframe :)
df = data.loc[:, labels].copy()
regions = df['region'].unique()

plots = [[1,1], [1,2], [2,1], [2,2]]

for element in labels[1:]:
  for index, region in enumerate(regions):
    i = 220 + index + 1
    plt.subplot(i)
    #sns.distplot(df[df['region'] == region][element], label=f'{region}-{element}', hist=False)
    plt.hist(df[df['region'] == region][element], label=f'{region}-{element}', bins=25, histtype='bar')
    if index == 0 or index == 2:
      plt.ylabel('Sample Count (n)')
    if index == 2 or index == 3:
      plt.xlabel('Concentration (PPM)')
    plt.legend()
    
    # Calculate EF
    ef = (df[element].mean()/df['Fe'].mean()) / (bg.loc[bg['type'] == 'Average'][element]/bg.loc[bg['type'] == 'Average']['Fe'])
    print(f'EF for {element} in {region}: {ef.values}')


    #print(f'===========Site: {region}===Element: {element} =======\n')
    #print(data.loc[data['region'] == region][element].describe())
    #print('==========================================\n')
    # Might move this to the outer most level loop to not have to keep constantly open a file
    # descriptor 
    # or save all txt to variablke and open and close file once after loop
    with open('vcp-raw-data-stats.txt', 'a') as f:
      f.write(f'===========Site: {region}===Element: {element} =======\n')
      f.write(str(data.loc[data['region'] == region][element].describe()) + '\n')
      f.write('==========================================\n')
     

  plt.suptitle(f'Sample Results for {element}')
  #plt.text(0.5,1.0, "Concentration (ppm)", ha="center", va="center")
  #plt.text(0.05,1.0, "Samples (n)", ha="center", va="center", rotation=90)
  plt.savefig(f'results/{element}.png')

  # clear plot
  plt.clf()

