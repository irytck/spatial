#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:37:45 2024

@author: user
"""

# Import libraries
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
import matplotlib.pyplot as plt
import contextily as ctx
from IPython.display import display, HTML

# Define the API url
api_url = "https://valencia.opendatasoft.com/api/records/1.0/search/"

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
        
        print("Green Zones GeoDataFrame extracted successfully")
    else:
        print("GeoDataFrame not created for Green Zones, geo_shape not available")
else: 
    print(f"Failed to retreive Green Zones data. Status code: {response_ev.status_code}")
    
# Parameteres for the Districts dataset API call
params_distrit = {
    "dataset": "districtes-distritos",  
    "rows": 30,  
    "fields": "objectid,nombre,coddistrit,geo_shape",
    "format": "json"
    }

# Api call for dist
response_distrit = requests.get(api_url, params=params_distrit)

# Check if succesful
if response_distrit.status_code == 200:
    # Parse the JSON data
    data_distrit = response_distrit.json()
    
    #Extract the relevant fields
    records_distrit = [record['fields'] for record in data_distrit['records']]
    dist = pd.DataFrame(records_distrit)
    
    # Convert 'geo_shape' column to Shapely geometry
    if 'geo_shape' in dist.columns:
        dist['geometry'] = dist['geo_shape'].apply(lambda x: shape(x) if x else None)
        
        # Convert df to gdf
        dist = gpd.GeoDataFrame(dist, geometry = 'geometry')
        
        dist.set_crs(epsg=4326, inplace=True)
        
        print("Districts GeoDataFrame extracted succesfully")
    else:
        print("GeoDataFrame not created for Districts, geo_shape not available")
else:
    print(f"Failed to retrieve Districts data. Status code: {response_distrit.status_code}")
    
    
# Load GeoData for sections
sections = gpd.read_file('/Users/user/projects/spatial/data/secciones_censales/secciones_censales.shp')
population = pd.read_excel('/Users/user/projects/spatial/data/population_2021.xlsx')
censo = pd.read_excel('/Users/user/projects/spatial/data/Censo_personas_2021.xls')
catastr = pd.read_excel('/Users/user/projects/spatial/data/valor_catastral_seccion.xlsx')
renta = pd.read_csv ('/Users/user/projects/spatial/data/renta_imputed_2015-2021.csv')

censo = censo.dropna()
sections = sections.dropna()

# Change column names
sections = sections.rename(columns = {'CODDISTSEC':'coddistsecc'})
censo = censo.rename(columns = {'CODDISTSEC':'coddistsecc'})
catastr = catastr.rename(columns = {'CODDISTSEC':'coddistsecc'})
renta = renta.rename(columns = {'CODDISTSEC':'coddistsecc'})

# According to "Directrices urbanas de AUMSA" Eliminate distrits 17-19

dist['coddistrit'].unique()
dist['coddistrit'] = dist['coddistrit'].astype(int)
dist = dist[dist['coddistrit'] != 17]
dist = dist[dist['coddistrit'] != 18]
dist = dist[dist['coddistrit'] != 19]

sections['CODDISTRIT']=sections['CODDISTRIT'].astype(int)
sections = sections[sections['CODDISTRIT'] != 17]
sections = sections[sections['CODDISTRIT'] != 18]
sections = sections[sections['CODDISTRIT'] != 19]

green_zones['dm'].unique()
green_zones = green_zones[green_zones['dm'] != 'POBLATS DEL NORD'] # distrit 17
green_zones = green_zones[green_zones['dm'] != 'POBLATS DE L`OEST'] #distrit 18 
green_zones = green_zones[green_zones['dm'] != 'POBLES DEL SUD'] #distrit 19

green_zones['tipologia'].unique()
green_zones = green_zones[green_zones['tipologia'] != 'Acompañamiento Viario']
green_zones = green_zones[green_zones['tipologia'] != 'Bulevar']

population = population[population['distrit'] != 19]
population = population[population['distrit'] != 18]
population = population[population['distrit'] != 17]

censo = censo[censo['CODDIST'] != 19]
censo = censo[censo['CODDIST'] != 18]
censo = censo[censo['CODDIST'] != 17]

catastr = catastr[catastr['CODDIST'] != 19]
catastr = catastr[catastr['CODDIST'] != 18]
catastr = catastr[catastr['CODDIST'] != 17]

renta['Distrito'] = renta['Distrito'].astype(int)
renta=renta[renta['Distrito'] != 19]
renta=renta[renta['Distrito'] != 18]
renta=renta[renta['Distrito'] != 17]

# Set 'coddistsecc' as primary key 
# Convert 'coddistsecc' into str with 4 digits 
sections['coddistsecc'] = sections['coddistsecc'].astype(int)
sections['coddistsecc'] = sections['coddistsecc'].astype(str).str.zfill(4)

censo['coddistsecc'] = censo['coddistsecc'].astype(int)
censo['coddistsecc'] = censo['coddistsecc'].astype(str).str.zfill(4)

catastr['coddistsecc'] = catastr['coddistsecc'].astype(str).str.zfill(4)

renta['coddistsecc'] = renta['coddistsecc'].astype(str).str.zfill(4)

population['coddistsecc'] = population['coddistsecc'].astype(str).str.zfill(4)

# Merge censo with poligons
census_tracts = pd.merge(sections, population[['coddistsecc', 'population']], on='coddistsecc', how='left')

# Create new variable population density
print(census_tracts.crs) # check crs, must be EPSG:25830
census_tracts['st_shape_area'] = census_tracts.geometry.area
census_tracts['population_density'] = census_tracts.apply(lambda x: x['population'] / (x['st_shape_area']), axis = 1)

# Add census data as income 2021, Vm2, education, age, % foreigners, % unemployed
census_tracts = pd.merge(census_tracts, censo[[
    'coddistsecc', 'edad_media', ' pct_extr', 'estudios_medios', 'pct_parados']],
    on='coddistsecc', how='left')
census_tracts = pd.merge(census_tracts, catastr[['coddistsecc', 'Valor por m2']],
                         on='coddistsecc', how='left')

census_tracts = pd.merge(census_tracts, renta[['coddistsecc', '2021']], on='coddistsecc', how='left')

# Add Distrit name to sections
census_tracts = census_tracts.rename(columns = {'CODDISTRIT':'coddistrit'})
census_tracts = pd.merge(census_tracts, dist[['coddistrit', 'nombre']], on='coddistrit', how='left')

census_tracts = census_tracts.rename(columns = {'nombre' : 'dm', 
                                                'Valor por m2' : 'Vm2',
                                                'edad_media' : 'age_avg',
                                                'estudios_medios' : 'education_avg',
                                                ' pct_extr' : 'pct_foreigners',
                                                'pct_parados' : 'pct_unemployed',
                                                '2021' : 'income_avg'})
census_tracts = census_tracts.drop(columns=['V1', 'V2', 'V3', 'V4'], errors='ignore')

# Add Distrit information to greenn zones df
dist = dist.rename(columns={'nombre':'dm'})

# Change names for Distrcts to ensure coherence
green_zones['dm'] = green_zones['dm'].replace({
    "SAIDIA": "LA SAIDIA",
    "L´EIXAMPLA": "L'EIXAMPLE",
    "PLA DEL REIAL" : "EL PLA DEL REAL",
    "OLIVERETA" : "L'OLIVERETA",
    "POBLATS MARÍTIMS" : "POBLATS MARITIMS"
})

# Merge
green_zones = pd.merge(green_zones, dist[['dm', 'coddistrit']], on = 'dm', how='left')

# "Park Turia" Doesn´t exist as a District in dist. Fill NaN in coddistrit column with "0" and convert to string

green_zones['coddistrit'] = green_zones['coddistrit'].fillna(0)
green_zones['coddistrit'] = green_zones['coddistrit'].astype(int)
green_zones['coddistrit'] = green_zones['coddistrit'].astype(str)

census_tracts = census_tracts.to_crs(epsg=4326)

# Plot census tracts with green zones together
fig, ax = plt.subplots(figsize = (20,20))
                           
census_tracts.plot(ax=ax, alpha=0.2, edgecolor = 'k', label='Census Tracts')
green_zones.plot(ax=ax, color = 'green', alpha=0.5, label='Green Zones')
    
# Add basemap
ctx.add_basemap(ax, crs=census_tracts.crs, source = ctx.providers.OpenStreetMap.Mapnik)
    
plt.show()

# Replace NaN values for green zone typology with 'Unknown'
tipologias = green_zones['tipologia'].fillna('Unknown').unique()

# Display information about the Green Zones GeoDataFrame
print(f"The Green Zones GeoDataFrame contains {green_zones.shape[0]} polygons, each representing a green area categorized into one of three typologies: {', '.join(map(str, tipologias))}")

# Display information about the Census tracts GeoDataFrame
print(f"\nThe Census tracts GeoDataFrame contains {census_tracts.shape[0]} polygons with {census_tracts.shape[1]} features.")

display(census_tracts.describe())


# Save as GeoJSON
output_path = "/Users/user/projects/spatial/data"

dist.to_file(f"{output_path}/districts.geojson", driver = "GeoJSON")
census_tracts.to_file(f"{output_path}/census_tracts.geojson", driver="GeoJSON")
green_zones.to_file(f"{output_path}/green_zones.geojson", driver="GeoJSON")
