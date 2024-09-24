#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Green Zone Accesibility calculation for each census tracts

"""

# Import libraries

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import shape
import matplotlib.pyplot as plt
import contextily as ctx    
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
import networkx as nx

# Open data
census_tracts = gpd.read_file('/Users/user/projects/spatial/data/census_tracts.geojson')
green_zones = gpd.read_file('/Users/user/projects/spatial/data/green_zones.geojson')

## Reproject crs
census_tracts = census_tracts.to_crs(epsg = 25830)
green_zones = green_zones.to_crs(epsg = 25830)

# Calculate centroid for census tracts
census_tracts['centroid'] = census_tracts.geometry.centroid

# Cuantify Accesibility to green zones based on buffer 100
'''
# Calcular el buffer de 500 metros
green_zones['buffer'] = green_zones.geometry.buffer(100)

# Crear un GeoDataFrame con los buffers
buffer = green_zones[['buffer']].copy()
buffer = buffer.set_geometry('buffer')

# Graficar los buffers
fig, ax = plt.subplots(figsize=(20, 20))
buffer.plot(ax=ax, color='green', alpha=0.1, label='Buffer')

ctx.add_basemap(ax, crs=buffer.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

plt.show()
'''


# Cuantify Accesibility to green zones gravitational weighted on Area
'''
# Green zones area
green_zones['area'] = green_zones.geometry.area

# Calculate distances

distances = {}

for idx_census, row_census in census_tracts.iterrows():
    centroid = row_census['centroid']
    distances[idx_census] = {}
    
    for idx_area, row_area in green_zones.iterrows():
        distance = centroid.distance(row_area.geometry)
        distances[idx_census][idx_area] = distance
        
# Apply gravitational model to evaluate the accesibility to green zones based 
census_tracts['A_j'] = 0.0

## Calculate A_j for each census tract
for idx_census, dist_dic in distances.items():
    A_j = 0
    for idx_area, distance in dist_dic.items():
        if distance > 0:
            A_j += green_zones.loc[idx_area, 'area'] / (distance ** 2)
            
    census_tracts.loc[idx_census, 'A_j'] = A_j 
 '''
 
# Cuantify Accesibility using Manhattan distances

'''def manhattan_distance(p1, p2):
    """Calculates Manhattan distance between two points: p1 and p2"""
    return abs(p1.x - p2.x) + abs(p1.y - p2.y)

distances = {}

# Iterate for each census tracts geometry
for idx_sec, row_sec in census_tracts.iterrows():
    centroide_sec = row_sec['centroid']
    distances[idx_sec] = {}
    
    # Iterar sobre cada área verde
    for idx_area, row_area in green_zones.iterrows():
        distancias_vertices = []
        
        # Check geometry Polygon or MultiPolygon
        if isinstance(row_area.geometry, Polygon):
            # Si es un polígono, iterar sobre los vértices
            for point in row_area.geometry.exterior.coords:
                # Calculate Manhattan distance from each censis centroid to the border of green area
                distancia = manhattan_distance(centroide_sec, Point(point))
                distancias_vertices.append(distancia)
                
        elif isinstance(row_area.geometry, MultiPolygon):
            # If MultiPolygon, iterate for each poygon
            for polygon in row_area.geometry.geoms:
                for point in polygon.exterior.coords:
                    distancia = manhattan_distance(centroide_sec, Point(point))
                    distancias_vertices.append(distancia)
        
        # Use minimal distance to the border
        if distancias_vertices:  # Solo si encontramos vértices
            distances[idx_sec][idx_area] = min(distancias_vertices)
            
 # Calculate Accesibility based on minimal Manhatan distances
census_tracts['Accessibility'] = 0.0

max_distance = 500  # meters

for idx_sec, dist_dict in distances.items():
    accessibility = 0
    for idx_area, distancia in dist_dict.items():
        if distancia <= max_distance:
            accessibility += 1
    census_tracts.loc[idx_sec, 'Accessibility'] = accessibility

# Plot Accesibility
fig, ax = plt.subplots(figsize=(20, 20))
green_zones.plot(ax=ax, color = 'green', label='Green Zones')
census_tracts.plot(column='Accessibility', ax=ax, legend=True, cmap='Greens', alpha = 0.5, legend_kwds={'label': "Accesibility to Green Zones"})
ctx.add_basemap(ax, crs=census_tracts.crs.to_string(), source=ctx.providers.CartoDB.Positron)
plt.title("Green Areas Accessibility in Valencia")
plt.show()
    '''

# Calculate Accesibility based on 2SFCA Method

 ## 1. Calculate distances

  ### 1.1 Define function for Manhattan distance calculation

def manhattan_distance(p1, p2):
    """Calculate the Manhattan distance between two points: p1 y p2"""
    return abs(p1.x - p2.x) + abs(p1.y - p2.y)

distances = {}

  ### 1.2 Perform calculation
   #### Iterate over each census tracts
for idx_sec, row_sec in census_tracts.iterrows():
    centroide_sec = row_sec['centroid']
    distances[idx_sec] = {}
    
    # Iterate over each green zone
    for idx_area, row_area in green_zones.iterrows():
        distancias_vertices = []
        
        # Check if the geometry Polygon o Multipolygon
        if isinstance(row_area.geometry, Polygon):
            # If it's a Polygon, calculate distance from each vertex to the census centroid
            for point in row_area.geometry.exterior.coords:
                distancia = manhattan_distance(centroide_sec, Point(point))
                distancias_vertices.append(distancia)
                
        elif isinstance(row_area.geometry, MultiPolygon):
            # If it's a MultiPolygon, iterate over each polygon and calculate distances
            for polygon in row_area.geometry.geoms:
                for point in polygon.exterior.coords:
                    distancia = manhattan_distance(centroide_sec, Point(point))
                    distancias_vertices.append(distancia)
        
        # Store minimum distance (since we're interested in the closest point)
        if distancias_vertices:  # Only store if vertices were found
            distances[idx_sec][idx_area] = min(distancias_vertices)

  ### 1.3 Visualize distances

   #### Select census tract and green zone
idx_sec = 1  # idx=1 census:0244
idx_area = 14  # idx=14: JARDIN DEL TURIA (TRAMO 10)

centroide_sec = census_tracts.loc[idx_sec, 'centroid']
area_geom = green_zones.loc[idx_area, 'geometry']

min_distance = float('inf') # Find the closest border
min_point = None

   #### If Polygon
if isinstance(area_geom, Polygon):
    for point in area_geom.exterior.coords:
        distancia = manhattan_distance(centroide_sec, Point(point))
        if distancia < min_distance:
            min_distance = distancia
            min_point = Point(point)

   #### If Multipolygon
elif isinstance(area_geom, MultiPolygon):
    for polygon in area_geom.geoms:
        for point in polygon.exterior.coords:
            distancia = manhattan_distance(centroide_sec, Point(point))
            if distancia < min_distance:
                min_distance = distancia
                min_point = Point(point)

line = LineString([centroide_sec, min_point]) # Line that represents manhattan distance

line_gdf = gpd.GeoDataFrame(geometry=[line]) #g gdf for the line

   #### Plot census tract, green zone and line
base = census_tracts.plot(color='lightblue', edgecolor='black', alpha=0.5, figsize=(8, 6))
green_zones.plot(ax=base, color='green', alpha=0.3)
line_gdf.plot(ax=base, color='red', linewidth=2)

   #### Plot point A and B
plt.scatter([centroide_sec.x], [centroide_sec.y], color='blue', label='Census Centroid', zorder=5)
plt.scatter([min_point.x], [min_point.y], color='orange', label='Nearest Green Area Point', zorder=5)

   #### Show the distance in meters
plt.text(0.05, 0.05,  # Normalized figure coordinates (5% from the left and bottom)
         f"{min_distance:.2f} meters", 
         color='red', fontsize=12, 
         transform=plt.gca().transAxes)
plt.legend()
plt.title(f"Manhattan Distance from Census Section {idx_sec} to Green Area {idx_area}")
plt.show()
    
 ## 2. Calculate weights

  ### 2.1 Define the distance-decay function G based on a Gaussian function

def G(d_ij, d_0=500):
    """Calculate the weight using the distance decay function G(d_ij, d_0)."""
    if d_ij <= d_0:
        return (np.exp(-0.5 * (d_ij / d_0)**2) - np.exp(-0.5)) / (1 - np.exp(-0.5))
    else:
        return 0

  ### 2.2 Calculate weights for each distance <= 1000 meters
d_0 = 1000  # Maximum threshold of 500 meters
weights = {}

   #### Iterate over the distances to calculate weights
for sec_idx, dist_dict in distances.items():
     # For each census tract, calculate weights for nearby green zones
    weights[sec_idx] = {area_idx: G(d_ij, d_0) for area_idx, d_ij in dist_dict.items() if d_ij <= d_0}

  ### 2.3 Plot distances and weights

   #### Select one census tract to visualize
sec_idx = 0

   #### Extract the distances and weights for the chosen census tract
distances_for_tract = distances[sec_idx]
weights_for_tract = {area_idx: G(d_ij, d_0=1000) for area_idx, d_ij in distances_for_tract.items()}

   #### Create lists for plotting
distances_list = list(distances_for_tract.values())
weights_list = list(weights_for_tract.values())

   #### Plot the distances vs weights
plt.figure(figsize=(8, 5))
plt.scatter(distances_list, weights_list, color='blue', label='Weight for each green area')
plt.plot(distances_list, weights_list, color='green', linestyle='--')
plt.title(f"Weights decay for census tract {sec_idx}")
plt.xlabel("Distance (meters)")
plt.ylabel("Weight")
plt.grid(True)
plt.legend()
plt.show()


 ## 3. Calculate R_i for each green zone

  ### Define the supply (S_i) as the area of green zones and demand (P_j) as population of census tracts
S_i = green_zones.geometry.area.tolist()
P_j = census_tracts['population'].tolist()

  ### 3.1 Define the function to calculate R_i (supply to demand ratio for green areas)
def Ri(S_i, weights, P_j):
    """Calculate R_i, the ratio of supply to demand for green areas."""
    total_weighted_population = sum(weight * P_j[idx] for idx, weight in weights.items())
    return S_i / total_weighted_population if total_weighted_population != 0 else 0 # Avoid division by zero

  ### 3.2 Perform caluclation (R_i for each green zone)
R_i = []

for area_idx, s in enumerate(S_i):
    # Gather the weights for each census tract that affects this green area
    area_weights = {sec_idx: weights[sec_idx][area_idx] for sec_idx in weights if area_idx in weights[sec_idx]}
    
    if area_weights:
        # Gather the population for the census tracts contributing to this area
        population_influence = {sec_idx: P_j[sec_idx] for sec_idx in area_weights.keys()}
        ratio = Ri(s, area_weights, population_influence)
        R_i.append(ratio)
    else:
        R_i.append(0)  # If no relevant areas are found, append 0

#### Output the calculated R_i values
print(R_i)

 ## 4. Calculate A_j for each census tract (Accessibility to green zones)
 
  ### 4.1 Define the function
def Aj(weights, ratios):
   """Calculate A_j, which is the accessibility for a given census tract, by summing the weighted R_i ratios."""
   weighted_ratios = [weight * ratio for weight, ratio in zip(weights, ratios)]
   return sum(weighted_ratios)

A_j_values = []

   #### List unique census tracts
census_unique = list(census_tracts.index)

   #### Merge distances and weights in df
dw_df = []
for sec_idx, area_distances in distances.items():
    for area_idx, dist in area_distances.items():
        dw_df.append({
            'Seccion': sec_idx,
            'Green_Zone': area_idx,
            'Distancia': dist,
            'Peso_G': G(dist, d_0)  # Calculamos el peso usando la función G previamente definida
        })
dw_df=pd.DataFrame(dw_df)

   #### Iterate over each census tract
for seccion in census_unique:
    # Filter distances corresponding to actual census tract 
    distance_seccion = dw_df[dw_df['Seccion'] == seccion]
    
    # Get ratios R_i that corresponds to a green zones associated to actual census tract
    ratios = [R_i[area_idx] for area_idx in distance_seccion['Green_Zone']]
    
    # Weights to list
    G_weights = distance_seccion['Peso_G'].tolist()  
    
    # Apply A_j function to calculate accesibility 
    A_j = Aj(G_weights, ratios)
    
    # Save result
    A_j_values.append(A_j)

print(A_j_values)
