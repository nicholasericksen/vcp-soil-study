import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import markdown
import json

sns.set()
pd.set_option('display.max_columns', 25)

data = pd.read_csv('vcp_soil_ppm.csv')

bg = pd.read_csv('vcp_bedrock_ppm.csv')


# Periodic Table
with open('periodic_table/PeriodicTableJSON.json', 'r') as f:
  periodic_table = json.load(f)




# Convert this whole idea into just a conditional clause such as
# " where mean > 1000 " to eliminate trace elements
labels = ['region', 'Mn', 'Fe', 'Ti', 'Ba', 'Ca', 'K', 'S', 'Si', 'Mg']
# can then eliminate copying the dataframe :)
df = data.loc[:, labels].copy()
regions = df['region'].unique()

plots = [[1,1], [1,2], [2,1], [2,2]]

ele_data = {}
for element in periodic_table['elements']:
  symbol = element['symbol']
  if symbol in labels:
    ele_data[symbol] = element 



md = '# Soil Report - VCP\n\n'
md += '### Background Soil Composition <br>\n'
md += str(bg.round(2))
md += '<br>\n'

efs = []

for element in labels[1:]:
  md += f'## {element}\n'
  md += f"**Summary**: {ele_data[element]['summary']} <br>\n"
  md += f"**Category**: {ele_data[element]['category']}<br>\n"
  md += f"**Number**: {ele_data[element]['number']}<br>\n"
  md += f"**Atomic Mass**: {ele_data[element]['atomic_mass']}<br>\n"
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
    
    md += f'### Region: {region}\n'
    #md += f'### Element: {element}\n'
    md += str(data.loc[data['region'] == region][element].describe().to_string()).replace('\n', '<br>')
    md += '\n'
    # Calculate EF
    ef = (df.loc[df['region'] == region][element].mean()/df.loc[df['region'] == region]['Fe'].mean()) / (bg.loc[bg['type'] == 'Average'][element]/bg.loc[bg['type'] == 'Average']['Fe'])
    efs.append(ef.values) 
    #print(f'EF for {element} in {region}: {ef.values}')
    md += f'#### EF: {ef.values}'
    md += '<br><br>\n'
  md += f"<br><img src='results/{element}.png' style='display:block;float:right;page-break-before:always'><br><br>\n"
  md += '<br><br>\n'

  plt.suptitle(f'Sample Results for {element}')
  plt.savefig(f'results/{element}.png')

  # clear plot
  plt.clf()

#md += data.describe().to_string().replace('\n', ' <br/> ') + '\n'
print(efs)
# Add EFs to dataframe
#df['ef'] = efs
## plt EF histogram
#for region in regions:
#  
#  plt.hist(df.loc[df['region'] == region]['ef'], label=region)
#
#plt.show()

with open('vcp-report.html', 'w+') as f:
  html = markdown.markdown(md)
  f.write(html)

with open('vcp-report.md', 'w+') as f:
  f.write(md)
