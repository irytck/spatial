# -*- coding: utf-8 -*-
"""
This notebook is to load data from APIs or other sources.
Once retrieved and cleaned the data, it saved locally in a CSV, GeoJSON format
"""

# Import libraries
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np

# Define the API url
api_url = "https://valencia.opendatasoft.com/api/records/1.0/search/"

# Parameteres for the sections dataset API call
params_census = {
    "dataset": "seccions-censals-secciones-censales",  
    "rows": 1000,  
    "fields": "coddistsecc,objectid,st_area_shape,coddistrit,geo_shape",
    "format": "json"
    }

# Api call for seccions_censals
response_census = requests.get(api_url, params=params_census)

# Check if succesful
if response_census.status_code == 200:
    # Parse the JSON data
    data_census = response_census.json()
    
    #Extract the relevant fields
    records_census = [record['fields'] for record in data_census['records']]
    df_census = pd.DataFrame(records_census)
    
    # Convert 'geo_shape' column to Shapely geometry
    if 'geo_shape' in df_census.columns:
        df_census['geometry'] = df_census['geo_shape'].apply(lambda x: shape(x) if x else None)
        
        # COnvert df to gdf
        census_tracts = gpd.GeoDataFrame(df_census, geometry = 'geometry')
        
        census_tracts.set_crs(epsg=4326, inplace=True)
        
        print("Census Tracts GeoDataFrame:")
        print(census_tracts.head())
    else:
        print("GeoDataFrame not created for Census Tracts, geo_shape not available")
else:
    print(f"Failed to retrieve Census Tracts data. Status code: {response_census.status_code}")
    
# Parameters for the espacios verdes API call
params_espai_verd = {
    "dataset" : "espais-verds-espacios-verdes",
    "rows" : 1000,
    "fields" : "objectid,id_jardin,nombre,barrio,tipologia,st_area_shape,zona,dm,ud_gestion,geo_shape",
    "format" : "json"
    }
# API call for espai_verd
response_ev = requests.get(api_url, params = params_espai_verd)

#Check if succesful
if response_ev.status_code == 200:
    # Parse json data
    data_espaiv = response_ev.json()
    
    #Extract relevant fields
    records_espaiv = [record['fields'] for record in data_espaiv['records']]
    
    #Convert to Df
    df_espaiv = pd.DataFrame(records_espaiv)
    
    #Convert 'geo_shape' to shapely geometry
    if 'geo_shape' in df_espaiv.columns:
        df_espaiv['geometry'] = df_espaiv['geo_shape'].apply(lambda x: shape(x) if x else None)
        
        # Convert Df to GDF
        green_zones = gpd.GeoDataFrame(df_espaiv, geometry = 'geometry')
        green_zones.set_crs(epsg=4326, inplace=True)
        
        print("Green Zones GeoDataFrame:")
        print(green_zones.head())
    else:
        print("GeoDataFrame not created for Green Zones, geo_shape not available")
else: 
    print(f"Failed to retreive Green Zones data. Status code: {response_ev.status_code}")
    

# According to "Directrices urbanas de AUMSA" Eliminate distrits 17-19

census_tracts['coddistrit']=census_tracts['coddistrit'].astype(int)
census_tracts = census_tracts[census_tracts['coddistrit'] != 17]
census_tracts = census_tracts[census_tracts['coddistrit'] != 18]
census_tracts = census_tracts[census_tracts['coddistrit'] != 19]

census_tracts = census_tracts.dropna()

green_zones['dm'].unique()
green_zones = green_zones[green_zones['dm'] != 'POBLATS DEL NORD'] # distrit 17
green_zones = green_zones[green_zones['dm'] != 'POBLATS DE L`OEST'] #distrit 18 
green_zones = green_zones[green_zones['dm'] != 'POBLES DEL SUD'] #distrit 19

green_zones['tipologia'].unique()
green_zones = green_zones[green_zones['tipologia'] != 'Acompañamiento Viario']
green_zones = green_zones[green_zones['tipologia'] != 'Bulevar']

# Plot census tracts and green zones together

if 'geometry' in census_tracts.columns and 'geometry' in green_zones.columns:
    fig, ax = plt.subplots(figsize = (20,20))
                           
    census_tracts.plot(ax=ax, alpha=0.2, edgecolor = 'k', label='Census Tracts')
    green_zones.plot(ax=ax, color = 'green', alpha=0.5, label='Green Zones')
    
    # Add basemap
    ctx.add_basemap(ax, crs=census_tracts.crs.to_string(), source = ctx.providers.OpenStreetMap.Mapnik)
    
    plt.show()
    
# Add population to census tracts

population = pd.read_excel('/Users/user/projects/spatial/data/Censo_2021.xlsx')
# Convert population['coddistsecc'] to string, remove '.0', and pad with leading zeros
population['coddistsecc'] = population['coddistsecc'].astype(str).str.zfill(4)

population['distrit'].unique()
population = population[population['distrit'] != 17]
population = population[population['distrit'] != 18]
population = population[population['distrit'] != 19]

# Merge
census_tracts = census_tracts.merge(population[['coddistsecc', 'population']], on='coddistsecc', how='left')

# Save as GeoJSON
output_path = "/Users/user/projects/spatial/data"
census_tracts.to_file(f"{output_path}/census_tracts.geojson", driver="GeoJSON")
green_zones.to_file(f"{output_path}/green_zones.geojson", driver="GeoJSON")