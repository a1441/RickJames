# RickJames

## 1. Import the relevant libraries

# Fuck warnings, live life dangerously
import warnings
warnings.filterwarnings("ignore")

# Idk if I even don't need pandas and numpy, that shit is getting imported
import pandas as pd
import numpy as np

# Visualize me daddy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Sklearn, thank God you exist.
from sklearn.cluster import KMeans
from sklearn import preprocessing

# Import ElbowVisualizer for the K means
from yellowbrick.cluster import KElbowVisualizer

## 2. Load the data

#European Monitoring Centre for Drugs and Drug Addiction
data = pd.read_excel('Rick James.xlsx')

![image.png](attachment:image.png)

data.head() #aaaayyyy lmao, we got NaNs

## 3. Transform the data

# Drop missing data
dataRickJames = data.dropna()

# Set index to the country that is being analyzed
dataRickJames = dataRickJames.set_index('Country')

# Round the values in the data for good measure
dataRickJames = dataRickJames.round(2)

# Select the dataframe and slice it to keep the the 3rd & 4th columns
dataRickJames = dataRickJames.iloc[:,2:] 

dataRickJames

### 4. Plot the data
Let's Czech it out!

# Scatter me daddy
plt.scatter(dataRickJames['2017 (Mode of Price)'],dataRickJames['2017 (Mode of Purity/Potency)'])

# Name ze axes
plt.xlabel('Price')
plt.ylabel('Purity / Potency')

## 5. Scale the data

# Create the X variables
X = dataRickJames[['2017 (Mode of Price)', '2017 (Mode of Purity/Potency)']]

# Scale me daddy, scale me hard.
X_scaled = preprocessing.scale(X)

#Seems Legit
X_scaled

## 6. Calculating the number of clusters with the Elbow Method

# List emptier than my head - Within-Cluster-Sum-of-Squares (wcss)
wcss =[] 

# Loop from 1 to length of the data
for c in range(1,len(dataRickJames)):
    
    # K stands for Clusters ROFL
    kmeans = KMeans(c)
    
    # Fitting the model harder than cement
    kmeans.fit(X_scaled)
    
    # Into the rabbit hole
    wcss.append(kmeans.inertia_)
    
# Eskeeeeeeeetit
wcss

## 7. Ploting the Elbow Method

# I love to do it with models 
model = KMeans()

# k is the lenght of the data, Yellowbrick has really easy models
visualizer = KElbowVisualizer(model, k=(range(1,len(dataRickJames))), timings=False)

# Fit the data to the visualizer
visualizer.fit(X_scaled)

# Finalize and render the figure
visualizer.show()        

## 8. Clusterize

# We are using the optimal number of K-means
kmeans_RJ = KMeans(4)

# Fit the data
kmeans_RJ.fit(X_scaled)

# Assign the clusters to the data
dataRickJames['Assigned cluster'] = kmeans_RJ.fit_predict(X_scaled)

## 9. Visualize and assess

# Sorting the data by clusters
dataRickJames = dataRickJames.sort_values(by='Assigned cluster', ascending=False)

# Models on the plot
plt.scatter(dataRickJames['2017 (Mode of Price)'],
            dataRickJames['2017 (Mode of Purity/Potency)'],
            c = dataRickJames['Assigned cluster'],cmap='rainbow')

# Name ze axes
plt.xlabel('Price')
plt.ylabel('Purity / Potency')

## 10. Select destinaction for vacation

# This task could have been solved with 2 lines of code, but I wouldn't have had a video
dataRickJames['Price2Potency'] = dataRickJames['2017 (Mode of Purity/Potency)'] / dataRickJames['2017 (Mode of Price)']

# Sort by price casue I'm poor like that
dataRickJames.sort_values(by='Price2Potency', ascending=False)



