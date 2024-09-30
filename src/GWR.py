#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:20:19 2024

@author: user
"""

# Import libraries
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from pysal.lib import weights
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from scipy.stats import norm
from scipy.stats import shapiro
from esda.moran import Moran

# Open data
data = gpd.read_file('/Users/user/projects/spatial/data/lisa.geojson')

features = ['population_density','education_avg','Vm2',
'income_avg']

X = data[features]
y = data['acces_norm']

# create histogram
plt.figure(figsize=(10,6))
plt.hist(y, bins=60, density=True, alpha=0.5, color='g', edgecolor='black')

# Adjust a curve for normal distribution
mu, std = norm.fit(y)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)

title = "Histogramm of Accesibility index adjusted to normal distribution"
plt.title(title)
plt.xlabel('acces_raw')
plt.ylabel('Density')

plt.grid(True)
plt.show()

'''
The skewness is a clue about potential spatial inequality in access to green zones. Investigate spatial clusters using Moran's I and local indicators 
of spatial association (LISA) to detect if high-accessibility areas are clustered in particular parts of the city, indicating inequitable distribution of green spaces.


Positively skewed distribution, indicates that most observations have lower accessibility to green zones, while a few areas have much higher accessibility.
Outliers or High-Access Areas: The long right tail of the distribution suggests that some areas have much better access to green spaces, potentially due to the proximity to large or multiple parks. 
These areas could be clustered in certain neighborhoods, leading to spatial inequality.

The positive skew suggests that the dependent variable (green zone accessibility) is not normally distributed. 
This could violate assumptions in some statistical models, though GWR is relatively robust to non-normality. 
However, extreme skewness may still affect the reliability of local coefficient estimates.
To reduce the skewness and improve model performance, considering log transformation.
First check residuals without log transformation if needed apply boxcox

Areas with higher accessibility might disproportionately influence the GWR results, especially in the spatial clusters of high-accessibility zones. 
This could lead to strong local relationships in some parts of the city (high-accessibility areas) and weaker or even insignificant relationships in low-accessibility areas.'''

data['centroid'] = data.geometry.centroid

# get coords from centroids
coords = np.array([(geom.x, geom.y) for geom in data['centroid']])

# Convert to array de NumPy
X = data[features].values
y = data['acces_norm'].values.reshape((-1,1))

# select Bandwidth
selector = Sel_BW(coords, y, X)
bw = selector.search()

# Train GWR Model
model = GWR(coords, y, X, bw)
results = model.fit()

# Model results
results.summary()

# Export coefficient
coefficients = results.params

# As reference, here is the (average) R2, AIC, and AICc
print('Mean R2 =', results.R2)
print('AIC =', results.aic)
print('AICc =', results.aicc)

# Add R2 to GeoDataframe
data['gwr_R2'] = results.localR2

# Map R2 Values
fig, ax = plt.subplots(figsize=(6, 6))
data.plot(column='gwr_R2', cmap = 'Greens', linewidth=0.01, scheme = 'JenksCaspall', k=7, legend=True, legend_kwds={'bbox_to_anchor':(1.10, 0.96)},  ax=ax)
ax.set_title('Local R2', fontsize=12)
ax.axis("off")
plt.show()

# Extract coefficients
data['gwr_intercept'] = results.params[:,0]
data['gwr_pop'] = results.params[:,1]
data['gwr_age'] = results.params[:,2]
data['gwr_vm2'] = results.params[:,3]
data['gwr_income'] = results.params[:,4]


# Filter t-values: standard alpha = 0.05
filtered_t = results.filter_tvals(alpha = 0.05)
# Filter t-values: corrected alpha due to multiple testing
filtered_tc = results.filter_tvals()
pd.DataFrame(filtered_tc)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20,20))

data.plot(column='gwr_pop', cmap = 'coolwarm', linewidth=0.05, scheme = 'FisherJenks', k=7,
          legend=False, legend_kwds={'bbox_to_anchor':(1.10, 0.96)},  ax=axes[0,0])
data[filtered_tc[:,1] == 0].plot(color='white', linewidth=0.05, edgecolor='black', ax=axes[0,0])

data.plot(column='gwr_age', cmap = 'coolwarm', linewidth=0.05, scheme = 'FisherJenks', k=7,
          legend=False, legend_kwds={'bbox_to_anchor':(1.10, 0.96)},  ax=axes[0,1])
data[filtered_tc[:,1] == 0].plot(color='white', linewidth=0.05, edgecolor='black', ax=axes[0,1])

data.plot(column='gwr_vm2', cmap = 'coolwarm', linewidth=0.05, scheme = 'FisherJenks', k=7,
          legend=False, legend_kwds={'bbox_to_anchor':(1.10, 0.96)},  ax=axes[1,0])
data[filtered_tc[:,1] == 0].plot(color='white', linewidth=0.05, edgecolor='black', ax=axes[1,0])

data.plot(column='gwr_income', cmap = 'coolwarm', linewidth=0.05, scheme = 'FisherJenks', k=7,
          legend=False, legend_kwds={'bbox_to_anchor':(1.10, 0.96)},  ax=axes[1,1])
data[filtered_tc[:,1] == 0].plot(color='white', linewidth=0.05, edgecolor='black', ax=axes[1,1])

plt.tight_layout()

axes[0,0].axis("off")
axes[0,1].axis("off")
axes[1,0].axis("off")
axes[1,1].axis("off")


axes[0,0].set_title('Population Density (BW: ' + str(bw) +'), significant coeffs', fontsize=12)
axes[0,1].set_title('Age Average  (BW: ' + str(bw) +'), significant coeffs', fontsize=12)
axes[1,0].set_title('Value m2  (BW: ' + str(bw) +'), significant coeffs', fontsize=12)
axes[1,1].set_title('Income Average  (BW: ' + str(bw) +'), significant coeffs', fontsize=12)

plt.show()

# Get the residuals
residuals = results.resid_response
data['gwr_residuals'] = residuals

# Weight matrix
w = weights.distance.Kernel.from_dataframe(data)

# Moran´s I
moran = Moran(data['gwr_residuals'], w)
print('Moran´s I:', moran.I)
print('p-value:', moran.p_sim)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
data.plot(column='gwr_residuals', cmap='coolwarm', linewidth=0.8, edgecolor='k', legend=True, ax=ax)
ax.set_title('Residuals of the Model GWR', fontsize=15)
ax.axis('off')
'''plt.text(-0.02, 0.4,
         f'Moran´s I: {moran.I:,.4f} \n'
        f'P-value: {moran.p_sim:,.2f}\n',
         fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.5))'''
plt.show()


# Normality Test Shapiro-Wilk
stat, p = shapiro(residuals)
print('Shapiro-Wilk:', stat)
print('p-value:', p)

# Histograma de los residuos
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Histogram Residuals for Model GWR')

plt.text(-0.2, 70,
         f'Shapiro-Wiki stat: {stat:,.4f} \n'
         f'P-value: {p:,.2f}\n',
         fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.5))
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Get predicted values
predicted = results.predy

# Plot residuos vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(predicted, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.show()




