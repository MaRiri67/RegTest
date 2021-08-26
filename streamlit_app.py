import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib
import matplotlib.pyplot as plt

st.title('First load dataset!')

raw_df = pd.read_csv('https://raw.githubusercontent.com/Rajkap/EDA_on_Would_countries_data/main/world-countries-data-analysis/proj_data/countries%20of%20the%20world.csv')
df = raw_df.copy()
df = df.drop('Climate', axis=1)
df['Region'] = df['Region'].str.strip()
asia_df = df[df['Region'].str.contains('ASIA')]
country_by_region = df.Region.value_counts()
country_by_region

# sns.set_style('darkgrid')
matplotlib.rcParams['font.size']=14
matplotlib.rcParams['figure.figsize']= (12,12)
matplotlib.rcParams['figure.facecolor']= '#00000000'

fig,ax = plt.subplots()

# sns.set(rc={'figure.figsize':(10,5)})
# ax = sns.barplot(country_by_region.index,country_by_region)
ax.bar(country_by_region.index,country_by_region)
ax.set_xlabel('Region')
ax.set_ylabel('Count')
plt.title('Number of countries in each region')
plt.xticks(rotation=85)
for i in ax.patches:
    width = i.get_width()
    height = i.get_height()
    ax.text(x= i.get_x() + width/2,
            y= height,
            s='{:.0f}'.format(height),
            ha='center')

st.pyplot(fig)

region_population = df.groupby('Region').Population.sum()
region_population

colors=['gold', 'red', 'lightcoral', 'lightskyblue','yellowgreen', 'gold', 'lightskyblue', 'lightcoral','blue','orange','silver']
explode=(0.2,0.5,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2)
fig,ax = plt.subplots()
ax.pie(region_population,explode=explode,labels=region_population.index,colors=colors,startangle=270,autopct='%1.2f%%')
plt.xticks(rataion=90)
st.pyplot(fig)
