'''Calculation of 2SFCA based on Population density P_j = population/section area)
'''
# Import libraries
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
import matplotlib.pyplot as plt
import contextily as ctx
import pickle
import os

# Open data
census_tracts = gpd.read_file('/Users/user/projects/spatial/data/census_tracts.geojson')
green_zones = gpd.read_file('/Users/user/projects/spatial/data/green_zones.geojson')

# Reproject crs
census_tracts = census_tracts.to_crs(epsg=25830)
green_zones = green_zones.to_crs(epsg=25830)

# Calculate centroid for census tracts
census_tracts['centroid'] = census_tracts.geometry.centroid
census_tracts['pop_density'] = census_tracts.apply(
    lambda row: row['population']/(row['st_area_shape']), axis=1)

# Distance calculations
# %% [fold]
# File to save precomputed distances
distances_file = 'recomputed_distances.pkl'

# Function to calculate Manhattan distances
def manhattan_distance(p1, p2):
    """Calculates Manhattan distance between two points: p1 and p2."""
    return abs(p1.x - p2.x) + abs(p1.y - p2.y)

# Load or calculate distances
if os.path.exists(distances_file):
    print("Loading precomputed distances...")
    with open(distances_file, 'rb') as f:
        distances = pickle.load(f)
else:
    print("Calculating distances...")
    # Calculate distances if file does not exist
    distances = {}

    for idx_sec, row_sec in census_tracts.iterrows():
        centroid_sec = row_sec['centroid']
        distances[idx_sec] = {}
        
        for idx_area, row_area in green_zones.iterrows():
            distancias_vertices = []
            
            if isinstance(row_area.geometry, Polygon):
                for point in row_area.geometry.exterior.coords:
                    distancia = manhattan_distance(centroid_sec, Point(point))
                    distancias_vertices.append(distancia)
                    
            elif isinstance(row_area.geometry, MultiPolygon):
                for polygon in row_area.geometry.geoms:
                    for point in polygon.exterior.coords:
                        distancia = manhattan_distance(centroid_sec, Point(point))
                        distancias_vertices.append(distancia)
            
            if distancias_vertices:  # Only store if vertices were found
                distances[idx_sec][idx_area] = min(distancias_vertices)
    
    # Save the computed distances to file
    with open(distances_file, 'wb') as f:
        pickle.dump(distances, f)
    print(f"Distances saved to {distances_file}")
    # %% [unfold]        

# Plot Distance example
# %% [fold]
# Plot a specific census tract and green zone
idx_sec = 1  # Change to relevant index
idx_area = 14  # Change to relevant index

centroid_sec = census_tracts.loc[idx_sec, 'centroid']
area_geom = green_zones.loc[idx_area, 'geometry']

# Find closest point and distance
min_distance = float('inf')
min_point = None

if isinstance(area_geom, Polygon):
    for point in area_geom.exterior.coords:
        distancia = manhattan_distance(centroid_sec, Point(point))
        if distancia < min_distance:
            min_distance = distancia
            min_point = Point(point)

elif isinstance(area_geom, MultiPolygon):
    for polygon in area_geom.geoms:
        for point in polygon.exterior.coords:
            distancia = manhattan_distance(centroid_sec, Point(point))
            if distancia < min_distance:
                min_distance = distancia
                min_point = Point(point)

# Create line representing Manhattan distance
line = LineString([centroid_sec, min_point])
line_gdf = gpd.GeoDataFrame(geometry=[line])

# Plot
base = census_tracts.plot(color='lightblue', edgecolor='black', alpha=0.5, figsize=(8, 6))
green_zones.plot(ax=base, color='green', alpha=0.3)
line_gdf.plot(ax=base, color='red', linewidth=2)

# Plot points
plt.scatter([centroid_sec.x], [centroid_sec.y], color='blue', label='Census Centroid', zorder=5)
plt.scatter([min_point.x], [min_point.y], color='orange', label='Nearest Green Area Point', zorder=5)

plt.text(0.05, 0.05, f"{min_distance:.2f} meters", color='red', fontsize=12, transform=plt.gca().transAxes)
plt.legend()
plt.title(f"Manhattan Distance from Census Section {idx_sec} to Green Area {idx_area}")
plt.show()

# %% [unfold]

# Weights Calculation
# %% [fold]
# Define Gaussian distance-decay function
def G(d_ij, d_0=500):
    """Distance-decay function G(d_ij, d_0) based on a Gaussian distribution."""
    if d_ij <= d_0:
        return (np.exp(-0.5 * (d_ij / d_0)**2) - np.exp(-0.5)) / (1 - np.exp(-0.5))
    else:
        return 0

# Calculate weights for each census tract and green area
d_0 = 500  # Max threshold for distance in meters
weights = {}

for sec_idx, dist_dict in distances.items():
    weights[sec_idx] = {area_idx: G(d_ij, d_0) for area_idx, d_ij in dist_dict.items() if d_ij <= d_0}

# Plot distances vs weights for one census tract
sec_idx = 0  # Change to desired index
distances_for_tract = distances[sec_idx]
weights_for_tract = {area_idx: G(d_ij, d_0=500) for area_idx, d_ij in distances_for_tract.items()}

distances_list = list(distances_for_tract.values())
weights_list = list(weights_for_tract.values())

plt.figure(figsize=(8, 5))
plt.scatter(distances_list, weights_list, color='blue', label='Weight for each green area')
plt.plot(distances_list, weights_list, color='green', linestyle='--')
plt.title(f"Weights decay for census tract {sec_idx}")
plt.xlabel("Distance (meters)")
plt.ylabel("Weight")
plt.grid(True)
plt.legend()
plt.show()
# %% [unfold]

# Calculate supply-to-demand ratio (R_i) for green zones
# %% [fold]
S_i = green_zones.geometry.area.tolist()  # Green zone areas
P_j = census_tracts['pop_density'].tolist()  # Population of census tracts

def Ri(S_i, weights, P_j):
    """Calculate supply-to-demand ratio for green areas (R_i)."""
    total_weighted_population = sum(weight * P_j[idx] for idx, weight in weights.items())
    return S_i / total_weighted_population if total_weighted_population != 0 else 0

R_i = []
for area_idx, s in enumerate(S_i):
    area_weights = {sec_idx: weights[sec_idx][area_idx] for sec_idx in weights if area_idx in weights[sec_idx]}
    
    if area_weights:
        population_influence = {sec_idx: P_j[sec_idx] for sec_idx in area_weights.keys()}
        ratio = Ri(s, area_weights, population_influence)
        R_i.append(ratio)
    else:
        R_i.append(0)
# %% [unfold]    

# Calculate A_j for each census tract (Accessibility to green zones)
# %% [fold]
def Aj(weights, ratios):
    """Calculate A_j, accessibility for a given census tract."""
    weighted_ratios = [weight * ratio for weight, ratio in zip(weights, ratios)]
    return sum(weighted_ratios)

A_j_values = []

census_unique = list(census_tracts.index)

dw_df = []
for sec_idx, area_distances in distances.items():
    for area_idx, dist in area_distances.items():
        dw_df.append({
            'Seccion': sec_idx,
            'Green_Zone': area_idx,
            'Distancia': dist,
            'Peso_G': G(dist, d_0)  # Use G function for weight calculation
        })
dw_df = pd.DataFrame(dw_df)

for seccion in census_unique:
    distance_seccion = dw_df[dw_df['Seccion'] == seccion]
    ratios = [R_i[area_idx] for area_idx in distance_seccion['Green_Zone']]
    G_weights = distance_seccion['Peso_G'].tolist()
    A_j = Aj(G_weights, ratios)
    A_j_values.append(A_j)

print(A_j_values)

census_tracts['acces_raw'] = A_j_values
# %% [unfold]

'''# Adjust accesibility based on census area
census_tracts['weighted_acces'] = census_tracts.apply(
    lambda row: row['acces_raw']*(1/np.sqrt(row['st_area_shape'])), axis=1)'''
                                     

from sklearn.preprocessing import MinMaxScaler

# Normalize 'Accesibility' btw 0 y 1 using Min-Max Scaling
scaler = MinMaxScaler()
census_tracts['acces_raw_norm'] = scaler.fit_transform(census_tracts[['acces_raw']])

'''# Normalize 'Accesibility' btw 0 y 1 using Min-Max Scaling
scaler = MinMaxScaler()
census_tracts['weighted_acces_norm'] = scaler.fit_transform(census_tracts[['weighted_acces']])'''


# Plot Accesibility Normalized
fig, ax = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw={'width_ratios': [1, 1]})

census_tracts.plot(
    column='acces_raw_norm', cmap='Greens', alpha=0.7, 
    edgecolor='grey', linewidth = 0.1, ax=ax[0], 
    legend=True, legend_kwds={'shrink': 0.6, 'orientation': 'horizontal', 'pad': 0.02})
ctx.add_basemap(ax[0], crs=census_tracts.crs, source=ctx.providers.CartoDB.Positron)
ax[0].set_title("Accesibility index 2SFCA (norm)")
ax[0].set_axis_off()

census_tracts.plot(
    column='acces_raw', cmap='Greens', alpha=0.7, 
    edgecolor='grey', linewidth = 0.1, ax=ax[1], 
    legend=True, legend_kwds={'shrink': 0.6, 'orientation': 'horizontal', 'pad': 0.02})
ctx.add_basemap(ax[1], crs=census_tracts.crs, source=ctx.providers.CartoDB.Positron)
ax[1].set_title("Raw Accesibility index 2SFCA")
ax[1].set_axis_off()

plt.tight_layout()
plt.show()

# Save as GeoJSON
census_tracts = census_tracts.drop(columns=['centroid'])
output_path = "/Users/user/projects/spatial/data"
census_tracts.to_file(f"{output_path}/Accesibility.geojson", driver="GeoJSON")