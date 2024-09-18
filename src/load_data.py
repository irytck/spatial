# -*- coding: utf-8 -*-
"""
First of all let´s load the Geo Data from City Hall Open Data Portal
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import pysal
import matplotlib.pyplot as plt
from pyproj import CRS
from pathlib import Path
import contextily as ctx
import osmnx as ox
import seaborn as sns
from pysal.lib import weights

#Imports to grab and parse data from the web
import requests
from io import StringIO
from shapely.geometry import shape

# Define the API URL
api_url = "https://valencia.opendatasoft.com/api/records/1.0/search/"

# Parameters for the API call
params = {
    "dataset": "seccions-censals-secciones-censales",  
    "rows": 1000,  
    "fields": "coddistsecc,objectid,st_area_shape,coddistrit,geo_shape",
    "format": "json"
}

# Make the API call
response = requests.get(api_url, params=params)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON data
    data = response.json()
    
    # Extract the relevant fields
    records = [record['fields'] for record in data['records']]
    
    # Convert to a DataFrame
    df = pd.DataFrame(records)
    
    # Convert 'geo_shape' to a Shapely geometry
    if 'geo_shape' in df.columns:
        df['geometry'] = df['geo_shape'].apply(lambda x: shape(x) if x else None)
        
        # Convert DataFrame to GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry='geometry')
        
        # Set the coordinate reference system
        gdf.set_crs(epsg=4326, inplace=True)  
        
        print(gdf.head())
        
        # Plot the geometries
        ax = gdf.plot(figsize = (10,10), alpha = 0.5, edgecolor = 'k')
        
        # Add basemap
        ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
        plt.show()
        
    else:
        print("GeoDataFrame not created, geo_shape not available.")
    
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")
