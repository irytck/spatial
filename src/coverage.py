#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 13:45:50 2024

@author: user
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

census = gpd.read_file('/Users/user/projects/spatial/data/census_tracts_2sfca.geojson')
parks = gpd.read_file('/Users/user/projects/spatial/data/green_zones.geojson')
districts = gpd.read_file('/Users/user/projects/spatial/data/districts.geojson')


# Calclulate % of Green Cover in the city
'''$\frac{Green cover in sq.km}{Municipal area in sq.km}*100$'''

'''Per capita availability of green cover is an important indicator to determine whether green cover is sufficient for each
person in the city. Zones in which population density is extremely high, appropriate measures need to be taken to make sure 
that adequate green cover is available per person. Please refer Standards of World Health Organisation, 
which prescribes minimum 9 sqm green cover per person and an ideal of 50 sqm per person.'''

parks = parks.to_crs(epsg=25830)
census = census.to_crs(epsg=25830)

parks['tipologia'].unique()

# Convert the area from sqm to ha
parks['area_ha'] = parks['st_area_shape'] / 10000  # 1 ha = 10,000 sqm

# Group by category and sum
area_by_category = parks.groupby('tipologia')['st_area_shape'].sum().reset_index()

# Convert total area to ha
area_by_category['area_ha'] = area_by_category['st_area_shape'] / 10000

# 2 decimals
area_by_category['area_ha'] = area_by_category['area_ha'].apply(lambda x: f'{x:,.2f} ha')

print("\n")
print('Area by category') 
print(area_by_category[['tipologia', 'area_ha']])
print("\n")

# Calculate total green coverage in sq.km
total_green_area_sqm = parks['st_area_shape'].sum()
total_green_area_sqkm = total_green_area_sqm/ 1_000_000
print(f'Total Green Coverage: {total_green_area_sqkm:,.2f} sqkm')
print("\n")

# Calculate Municipal area in sq.km
mun_area_sqm = census['st_shape_area'].sum()
mun_area_sqkm = mun_area_sqm/1_000_000
print(f'Total Municipal Area: {mun_area_sqkm:,.2f} sqkm')
print("\n")

# Calculate % of the green cover in the city
green_cover_pct = (total_green_area_sqkm/mun_area_sqkm)*100
print(f'Percentage of green cover in the city: {green_cover_pct:,.2f} %')

# Spatial join btw parks and census
parks_with_census = gpd.sjoin(parks, census, how="left", op="intersects")

# Verify parks per section
parks_per_section = parks_with_census.groupby('coddistsecc').size()

# Aggregate the green area per zone. Group by the 'zona' column from parks and sum the green area
green_area_per_zone = parks_with_census.groupby('zona')['st_area_shape'].sum().reset_index()
green_area_per_zone['green_area_ha'] = green_area_per_zone['st_area_shape'] / 10000
green_area_per_zone['green_area_ha'] = green_area_per_zone['green_area_ha'].apply(lambda x: f'{x:,.2f} ha')

# Aggregate population per zone
population_per_zone = parks_with_census.groupby('zona')['population'].sum().reset_index()

data_by_zone = pd.merge(green_area_per_zone, population_per_zone, on='zona')

data_by_zone['sqm_per_person'] = data_by_zone['st_area_shape']/data_by_zone['population']

print("\n")
print("Green Cover sqm per person:")
print(data_by_zone[['zona','green_area_ha', 'sqm_per_person']])
print("\n")

# Plot
plt.figure(figsize=(10,6))
bars = plt.bar(data_by_zone['zona'], data_by_zone['sqm_per_person'], color='skyblue')

plt.axhline(y=50, color='green', linestyle='--', label='ideal 50 sqm by World Health Organisation')
plt.axhline(y=9, color='red', linestyle='--', label='minimum 9 sqm by World Health Organisation')

# Add text for area by category
for i, (zona, sqm) in enumerate(zip(data_by_zone['zona'], data_by_zone['sqm_per_person'])):
    plt.text(i, sqm + 1, f'{sqm:,.2f}', ha='center', va='bottom', fontsize=10)

# Add textbox with total green coverage and municipal 
plt.text(-0.5, 53,
         f'Total Green Coverage: {total_green_area_sqkm:,.2f} sqkm\n'
         f'Total Municipal Area: {mun_area_sqkm:,.2f} sqkm\n'
         f'Percentage Green Cover: {green_cover_pct:,.2f}%',
         fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.5))

plt.xlabel('Zone')
plt.ylabel('Green Cover (sqm per person)')
plt.title("Green Cover per person by Zone (sqm)")

plt.xticks(rotation=90)

plt.legend()

plt.tight_layout()
plt.show()

# Aggregate the green zones per District
green_area_per_district = parks_with_census.groupby('dm_right')['st_area_shape'].sum().reset_index()
green_area_per_district['green_area_ha'] = green_area_per_district['st_area_shape'] / 10000
green_area_per_district['green_area_ha'] = green_area_per_district['green_area_ha'].apply(lambda x: f'{x:,.2f} ha')

population_per_district = parks_with_census.groupby('dm_right')['population'].sum().reset_index()

data_by_district = pd.merge(green_area_per_district, population_per_district, on='dm_right')
data_by_district['sqm_per_person'] = data_by_district['st_area_shape']/data_by_district['population']

data_by_district = data_by_district.rename(columns = {'dm_right' : 'dm'})


# Barplot Green covergae by District

plt.figure(figsize=(10,6))
bars = plt.bar(data_by_district['dm'], data_by_district['sqm_per_person'], color='skyblue')

plt.axhline(y=50, color='green', linestyle='--', label='ideal 50 sqm by World Health Organisation')
plt.axhline(y=9, color='red', linestyle='--', label='minimum 9 sqm by World Health Organisation')

# Add text for area by category
for i, (zona, sqm) in enumerate(zip(data_by_district['dm'], data_by_district['sqm_per_person'])):
    plt.text(i, sqm + 1, f'{sqm:,.2f}', ha='center', va='bottom', fontsize=10)

plt.xlabel('District')
plt.ylabel('Green Cover (sqm per person)')
plt.title("Green Cover per person by District (sqm)")

plt.xticks(rotation=90)
plt.legend()

plt.tight_layout()
plt.show()

# Plot Covergae by District on the Map 
# Create GeoDataFrame
districts_coverage = pd.merge(data_by_district, districts[['dm', 'geometry']], on='dm', how='left')
districts_coverage = gpd.GeoDataFrame(districts_coverage, geometry='geometry')
districts_coverage.set_crs(epsg=4326, inplace=True)


fig, ax = plt.subplots(figsize=(20,20))

districts_coverage.plot(
    column='sqm_per_person', cmap='Blues', alpha=0.5, 
    edgecolor='grey', linewidth = 0.1, ax=ax, 
    legend=True, legend_kwds={'shrink': 0.6})
ctx.add_basemap(ax, crs=districts_coverage.crs, source=ctx.providers.CartoDB.Positron)
plt.title("Green Coverage per person by District (sqm)", size = 25)

plt.tight_layout()
plt.show()
