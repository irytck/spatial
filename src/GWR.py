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
from scipy import stats
from scipy.stats import norm
from scipy.stats import shapiro
from scipy.stats import skew
from esda.moran import Moran

# Open data
data = gpd.read_file('/Users/user/projects/spatial/data/lisa.geojson')

features = ['population_density','education_avg','income_avg']

X = data[features]
y = data['boxcox_accessibility']



# create histogram
plt.figure(figsize=(10,6))
plt.hist(y, bins=60, density=True, alpha=0.5, color='g', edgecolor='black')

# Adjust a curve for normal distribution
mu, std = norm.fit(data['acces_raw'])
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

# Calculate skewness using scipy
accessibility_skew = skew(data['acces_raw'], nan_policy='omit')

# Print the result
print(f"Skewness of accessibility index: {accessibility_skew}")

'''The positive skewness in the accessibility index distribution highlights spatial inequality in access to green spaces. 
We already detected these unequalities in local analysis. The long right tail of the distribution indicates the presence 
of outliers—areas with high accessibility due to proximity to large or multiple green spaces. 

The positive skew also implies that the dependent variable (green zone accessibility) deviates from a normal distribution, 
which could violate assumptions in statistical models. While Geographically Weighted Regression (GWR) is robust to moderate 
deviations from normality, extreme skewness might affect the reliability of local coefficient estimates. High-accessibility 
areas could disproportionately influence the GWR results, leading to stronger local relationships in those zones while 
producing weaker or insignificant relationships in low-accessibility areas.

To address the skewness and potentially improve model performance, square root transformation is 
applied to correct for the skew and stabilize variance.'''


# Shift the data by adding a small constant
data['acces_shifted'] = data['acces_raw'] + 0.001

# Apply Box-Cox transformation on the shifted data
#data['boxcox_accessibility'], _ = stats.boxcox(data['acces_shifted'])
data['sqrt_accessibility'] = np.sqrt(data['acces_norm_shifted'])

# Calculate skewness using scipy
accessibility_skew = skew(data['sqrt_accessibility'], nan_policy='omit')

# Print the result
print(f"Skewness of accessibility index: {accessibility_skew}")

# GWR

data['centroid'] = data.geometry.centroid

# get coords from centroids
coords = np.array([(geom.x, geom.y) for geom in data['centroid']])

# Convert to array de NumPy
X = data[features].values
y = data['sqrt_accessibility'].values.reshape((-1,1))

# select Bandwidth
selector = Sel_BW(coords, y, X)
bw = selector.search()

# Train GWR Model
model = GWR(coords, y, X, bw)
results = model.fit()

# Model results
summary_gwr = results.summary()

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
data['gwr_education'] = results.params[:,2]
data['gwr_income'] = results.params[:,3]


# Filter t-values: standard alpha = 0.05
filtered_t = results.filter_tvals(alpha = 0.05)
# Filter t-values: corrected alpha due to multiple testing
filtered_tc = results.filter_tvals()
pd.DataFrame(filtered_tc)

# PLot Coeficients
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20,20))

data.plot(column='gwr_intercept', cmap = 'coolwarm', linewidth=0.05, scheme = 'FisherJenks', k=7,
          legend=False, legend_kwds={'bbox_to_anchor':(1.10, 0.96)},  ax=axes[0,0])
data[filtered_tc[:,1] == 0].plot(color='white', linewidth=0.05, edgecolor='black', ax=axes[0,0])

data.plot(column='gwr_pop', cmap = 'coolwarm', linewidth=0.05, scheme = 'FisherJenks', k=7,
          legend=False, legend_kwds={'bbox_to_anchor':(1.10, 0.96)},  ax=axes[0,1])
data[filtered_tc[:,1] == 0].plot(color='white', linewidth=0.05, edgecolor='black', ax=axes[0,1])

data.plot(column='gwr_education', cmap = 'coolwarm', linewidth=0.05, scheme = 'FisherJenks', k=7,
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


axes[0,0].set_title('Intercept (BW: ' + str(bw) +'), significant coeffs', fontsize=12)
axes[0,1].set_title('Population Density  (BW: ' + str(bw) +'), significant coeffs', fontsize=12)
axes[1,0].set_title('Education Average  (BW: ' + str(bw) +'), significant coeffs', fontsize=12)
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

#plt.text(-0.02, 0.4,
         #f'Moran´s I: {moran.I:,.4f} \n'
        #f'P-value: {moran.p_sim:,.2f}\n',
         #fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.5))
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



