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



# Cuantify Accesibility to green zones gravitational weighted on Area

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

 
# Cuantify Accesibility using Manhattan distances

def manhattan_distance(p1, p2):
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

