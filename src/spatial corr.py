#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 13:45:50 2024

@author: user
"""
# Import libraries
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
import osmnx as ox
import pysal
from pysal.lib import weights
import seaborn as sns
import esda
from splot import esda as esdaplot


census = gpd.read_file('/Users/user/projects/spatial/data/Accesibility.geojson')
parks = gpd.read_file('/Users/user/projects/spatial/data/green_zones.geojson')
dist = gpd.read_file('/Users/user/projects/spatial/data/districts.geojson')

# Construct Weight matrix based on Kernel function
'''Kernel weights measure the relationship between observations based on spatial proximity, 
where closer locations have a stronger connection than distant ones, and this relationship 
is modulated by a specific kernel function. These weights, used in spatial analysis, 
follow Tobler's First Law and allow for different sensitivities in how influence decreases with distance, 
adapting to various spatial contexts.'''

w = weights.distance.Kernel.from_dataframe(census, fixed=False, k=16) # I chose a maximum of k=16 neighbors to capture the full range of variability, as sections have between 1 and 16 neighbors, with a mode of 7. 
w.bandwidth[:5]

'''Adaptive bandwidth ensures that each census section has exactly 16 neighbors, 
adjusting the distance based on local neighbor density. 
This approach captures spatial variability more accurately, as densely populated areas use smaller bandwidths, 
while sparsely populated areas use larger ones.'''

# Plot weights
f, ax = plt.subplots(
    1,2, figsize = (10,5), subplot_kw=dict(aspect="equal")
    )
census.assign(w_0=w[0]).plot(
    "w_0", cmap='plasma', ax=ax[0], legend=True) # Append weights on 0 census and plot

census.assign(w_15 = w[15]).plot(
    "w_15", cmap='plasma', ax=ax[1], legend=True)

census.iloc[[0], :].centroid.plot(
    ax=ax[0], marker = "*", color='k', label="Focal Tract")

census.iloc[[15], :].centroid.plot(
    ax=ax[1], marker = "*", color = 'k', label = "Focal Tract")

ax[0].set_title(f"Kernel centered on census: {census.iloc[0]['coddistsecc']}")
ax[1].set_title(f"Kernel centered on census: {census.iloc[15]['coddistsecc']}")

[ax_.set_axis_off() for ax_ in ax]
[ax_.legend(loc='upper left') for ax_ in ax]

plt.show()

# Global Spatial Correlation

## Spatial Lag model
'''Spatial Lag helps to understand how the values of a variable, such as 2SFCA (Two-Step Floating Catchment Area),
in one section are influenced by the values in neighboring sections. It effectively captures the spatial dependence,
revealing how accessibility to green areas in one location is shaped by the surrounding areas, acting as a local 
"smoother" of the variable across the spatial context. $Y_{sl}=W*Y$'''

census['acces_lag'] = weights.spatial_lag.lag_spatial(w, census['acces_norm'])

# Choroplet map: Accesibility vs Accesibilty_Spatial Lag
'''Four classification methods were tested: Equal Intervals, Quantiles, Fisher-Jenks, and Jenks-Caspall. 
Ultimately, 𝑘=7 was selected for the Fisher-Jenks method as it struck the best balance between map complexity and 
the representation of significant data variations. The method's lowest ADCM score further indicated that it provided 
the most compact and accurate data classification.'''

f, ax = plt.subplots(1,2, figsize = (40,20))

census.plot(
    column = 'acces_norm',
    cmap = 'Greens',
    scheme = 'FisherJenks',
    k=5,
    edgecolor = 'white',
    linewidth = 0.05,
    alpha = 0.7,
    legend =True,
    ax=ax[0],
    legend_kwds = {'fontsize' : 20})

ctx.add_basemap(ax[0], crs=census.crs, source=ctx.providers.CartoDB.Positron)
ax[0].set_axis_off()
ax[0].set_title("Accesibility (2SFCA)", size = 25)

census.plot(
    column = 'acces_lag',
    cmap ='Greens',
    scheme = 'FisherJenks',
    k=5,
    edgecolor = 'white',
    linewidth = 0.05,
    alpha = 0.7,
    legend = True,
    ax=ax[1],
    legend_kwds = {'fontsize' : 20})

ctx.add_basemap(ax[1], crs=census.crs, source=ctx.providers.CartoDB.Positron)
ax[1].set_axis_off()
ax[1].set_title("Accesibility Lag (2SFCA)", size =25)

plt.suptitle("Choroplet Map (Clasification Algorithm: FisherJenks, k=7")
plt.tight_layout()
plt.show()

# Moran´s I
'''Moran's I indicates whether nearby locations exhibit similar values for a given variable such as green zone accesibility index.
The Moran Plot visually represents this relationship by plotting the standardized values of a variable against its spatial lag, 
helping to identify clusters or patterns of similarity in green zone accessibility across different regions. 
This analysis allows to detect spatial patterns of inequality in access to green areas'''

census['acces_std'] = census['acces_norm'] - census['acces_norm'].mean()
census['acces_lag_std'] = weights.lag_spatial(w, census['acces_std'])
mi_acces = esda.moran.Moran(census['acces_norm'], w)

resume_moran = pd.DataFrame({
    "Moran´s I" : [mi_acces.I],
    "p-value" : [mi_acces.p_sim.mean()]
    })
print(resume_moran)

# Moran Plot
fig, ax = plt.subplots(figsize = (8,8))
sns.regplot(
    x = 'acces_std',
    y = 'acces_lag_std',
    ci = None,
    data = census,
    line_kws={"color": "r"},
    ax=ax)
plt.axvline(0, c="k", alpha=0.5)
plt.axhline(0, c = "k", alpha = 0.5)

plt.text(0.01, 2.5,
         f"Moran´s I: {mi_acces.I:,.2f}\n"
         f"P-value: {mi_acces.p_sim.mean()}", size = 20, c='grey')

plt.title("Moran Plot", size = 25)

plt.tight_layout()
plt.show()

# Local Spatial Correlation
lisa = esda.moran.Moran_Local(census['acces_norm'], w)

fig, ax = plt.subplots(figsize = (20,20))
esdaplot.lisa_cluster(lisa, census, p = 0.05, ax=ax)
plt.text()
ctx.add_basemap(ax, crs=census.crs, source=ctx.providers.CartoDB.Positron)
ax.set_axis_off()
ax.set_title("LISA Cluster Map", size =25)

plt.tight_layout()
plt.show()

counts = pd.Series(lisa.q).value_counts()

print("Number of observations in each cluster: \n")
print(counts)

# Export significance levels
census['p_sim_acces'] = lisa.p_sim
# `1` if significant (at 5% confidence level), `0` otherwise
sig = 1 * (lisa.p_sim < 0.05)
census['sign_dummy'] = sig
# Labels from value
spots_labels = {
    0: "NS",
    1: "HH",
    2: "LH",
    3: "LL"
    }
spots = lisa.q * sig # Pick as part of a quadrant only significant polygons, assign `0` otherwise (Non-significant polygons)
census['local_labels'] = pd.Series(
    spots, index=census.index # initialise a Series using values and `db` index
    ).map(spots_labels) # map each value to corresponding label based on the `spots_labels` mapping

census.head()

# Save as GeoJSON
output_path = "/Users/user/projects/spatial/data"
census.to_file(f"{output_path}/lisa.geojson", driver="GeoJSON")
