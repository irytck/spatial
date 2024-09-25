#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 13:45:50 2024

@author: user
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx

census = gpd.read_file('/Users/user/projects/spatial/data/Accesibility.geojson')
parks = gpd.read_file('/Users/user/projects/spatial/data/green_zones.geojson')
dist = gpd.read_file('/Users/user/projects/spatial/data/districts.geojson')

# Global Spatial Correlation



# Local Spatial Correlation

