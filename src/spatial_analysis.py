"""
This script performs spatial autocorrelation analysis on census tract data related to accessibility to green spaces. Using geospatial and statistical libraries such as `geopandas`, `PySAL`, and `esda`, the script constructs a spatial weights matrix based on a kernel function, calculates spatial lag, and visualizes global and local spatial correlations through choropleth maps, Moran's I statistics, and LISA (Local Indicators of Spatial Association) analysis. The results help identify spatial clusters of similar or contrasting accessibility values across census tracts. The processed data is exported as GeoJSON for further analysis or visualization.

Author: irytck
Date Created: September 30, 2024
"""
# Import libraries
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from pysal.lib import weights
import seaborn as sns
import esda
from splot import esda as esdaplot


census_tracts = gpd.read_file('../data/census_tracts_2sfca.geojson')

# Construct Weight matrix based on Kernel function
w = weights.distance.Kernel.from_dataframe(census_tracts, fixed=False, k=16) # I chose a maximum of k=16 neighbors to capture the full range of variability, as sections have between 1 and 16 neighbors, with a mode of 7. 
w.bandwidth[:5]

# Plot weights
f, ax = plt.subplots(
    1,2, figsize = (10,5), subplot_kw=dict(aspect="equal")
    )
census_tracts.assign(w_0=w[0]).plot(
    "w_0", cmap='plasma', ax=ax[0], legend=True) # Append weights on 0 census and plot

census_tracts.assign(w_100 = w[100]).plot(
    "w_100", cmap='plasma', ax=ax[1], legend=True)

census_tracts.iloc[[0], :].centroid.plot(
    ax=ax[0], marker = "*", color='k', label="Focal Tract")

census_tracts.iloc[[100], :].centroid.plot(
    ax=ax[1], marker = "*", color = 'k', label = "Focal Tract")

ax[0].set_title(f"Kernel centered on census: {census_tracts.iloc[0]['coddistsecc']}")
ax[1].set_title(f"Kernel centered on census: {census_tracts.iloc[35]['coddistsecc']}")

[ax_.set_axis_off() for ax_ in ax]
[ax_.legend(loc='upper left') for ax_ in ax]

plt.suptitle("Comparative Visualization of Kernel Weights Across Two Census Tracts", fontsize = 25)
plt.show()

# Global Spatial Correlation

## Spatial Lag model
census_tracts['acces_lag'] = weights.spatial_lag.lag_spatial(w, census_tracts['acces_norm'])

# Choroplet map: Accesibility vs Accesibilty_Spatial Lag
'''Four classification methods were tested: Equal Intervals, Quantiles, Fisher-Jenks, and Jenks-Caspall. Ultimately, 𝑘=7 was selected for the Fisher-Jenks method as it struck the best balance between map complexity and the representation of significant data variations. The method's lowest ADCM score further indicated that it provided the most compact and accurate data classification.'''

f, ax = plt.subplots(1, 2, figsize=(20, 10))

# Plot the index
census_tracts.plot(
    column='acces_norm',
    cmap='Greens',
    scheme='FisherJenks',
    k=5,
    edgecolor='white',
    linewidth=0.05,
    alpha=0.7,
    legend=True,
    ax=ax[0],
    legend_kwds={'fontsize': 20}
)

ctx.add_basemap(ax[0], crs=census_tracts.crs, source=ctx.providers.CartoDB.Positron)
ax[0].set_axis_off()
ax[0].set_title("Accessibility Index", size=15)

# Plot the spatial lag
census_tracts.plot(
    column='acces_lag',
    cmap='Greens',
    scheme='FisherJenks',
    k=5,
    edgecolor='white',
    linewidth=0.05,
    alpha=0.7,
    legend=True,
    ax=ax[1],
    legend_kwds={'fontsize': 20}
)

ctx.add_basemap(ax[1], crs=census_tracts.crs, source=ctx.providers.CartoDB.Positron)
ax[1].set_axis_off()
ax[1].set_title("Accessibility Index - Spatial Lag Model", size=15)

# Set the main title
plt.suptitle("Choropleth Maps of Accessibility: Raw Index vs. Spatial Lag Model", fontsize=25)

# Set the subtitle
plt.figtext(0.5, 0.92, "Classification Algorithm: Fisher-Jenks, k=5", ha='center', fontsize=20, fontstyle='italic')

# Adjust layout to prevent overlap
plt.subplots_adjust(top=0.85)  # Adjust top value

plt.tight_layout(rect=[0, 0, 1, 0.85])  # Adjust rect to give room for the title and subtitle
plt.show()

# Moran´s I
census_tracts['acces_std'] = census_tracts['acces_norm'] - census_tracts['acces_norm'].mean()
census_tracts['acces_lag_std'] = weights.lag_spatial(w, census_tracts['acces_std'])
mi_acces = esda.moran.Moran(census_tracts['acces_norm'], w)

resume_moran = pd.DataFrame({
    "Moran´s I" : [mi_acces.I],
    "p-value" : [mi_acces.p_sim.mean()]
    })
print(f"Moran´s I statistics\n")
display(resume_moran)

# Moran Plot
fig, ax = plt.subplots(figsize = (10,10))
sns.regplot(
    x = 'acces_std',
    y = 'acces_lag_std',
    ci = None,
    data = census_tracts,
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
lisa = esda.moran.Moran_Local(census_tracts['acces_norm'], w)

fig, ax = plt.subplots(figsize = (10,10))
esdaplot.lisa_cluster(lisa, census_tracts, p = 0.05, ax=ax)
ctx.add_basemap(ax, crs=census_tracts.crs, source=ctx.providers.CartoDB.Positron)
ax.set_axis_off()
ax.set_title("LISA Cluster Analysis: Mapping HH and LL Accessibility Index Values", size =25)

plt.tight_layout()
plt.show()

counts = pd.Series(lisa.q).value_counts()

print("Number of observations in each cluster: \n")
display(counts)

# Export significance levels
census_tracts['p_sim_acces'] = lisa.p_sim
# `1` if significant (at 5% confidence level), `0` otherwise
sig = 1 * (lisa.p_sim < 0.05)
census_tracts['sign_dummy'] = sig
# Labels from value
spots_labels = {
    0: "NS",
    1: "HH",
    2: "LH",
    3: "LL"
    }
spots = lisa.q * sig # Pick as part of a quadrant only significant polygons, assign `0` otherwise (Non-significant polygons)
census_tracts['local_labels'] = pd.Series(
    spots, index=census_tracts.index # initialise a Series using values and `db` index
    ).map(spots_labels) # map each value to corresponding label based on the `spots_labels` mapping

# Save as GeoJSON
output_path = "../data"
census_tracts.to_file(f"{output_path}/lisa.geojson", driver="GeoJSON")
