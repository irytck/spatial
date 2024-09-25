# Green Coverage and Accessibility Analysis in Valencia

This project aims to analyze the availability and accessibility of green zones in the city of Valencia, Spain, using spatial analysis techniques. The analysis focuses on evaluating the per capita availability of green spaces, understanding disparities in green coverage, and correlating green coverage with land surface temperature.

## Project Overview

This study utilizes spatial data on census tracts and green zones to conduct a comprehensive analysis of green coverage and its accessibility in Valencia. The following are the key objectives:

- **Green Coverage Per Capita**: Analyzing the distribution of green spaces per person in each census tract, in accordance with the standards set by the World Health Organization (WHO).
- **Accessibility Index**: Calculating the accessibility to green areas using the 2-Step Floating Catchment Area (2SFCA) method, which accounts for population density, green area sizes, and distances between census tracts and green zones.
- **Hot Spot Analysis**: Identifying hot spots and cold spots of green coverage using spatial autocorrelation techniques.
- **Land Surface Temperature Correlation**: Performing geographically weighted regression (GWR) to analyze the relationship between green coverage and land surface temperature in different areas of the city.

## Data Sources

The data used for this project comes from the following sources:
- **Census and demographic data**: Retrieved from the [Instituto Nacional de Estadística](https://www.ine.es).
- **Green zones geospatial data**: Accessed via the [Geo Portal del Ayuntamiento de Valencia](https://valencia.opendatasoft.com/pages/home/) through an API.

## Methodology

The project is divided into several stages:

1. **Exploratory Green Coverage Analysis**:  
   - **Per Capita Availability of Green Space**: This metric is used to assess the adequacy of green spaces relative to the population density in each census tract. Areas with extremely high population density are identified, where additional green spaces may be required.

2. **Accessibility Analysis**:
   - **2SFCA Method**: Accessibility is measured using the 2-Step Floating Catchment Area (2SFCA) method, which factors in:
     - **Distances**: Manhattan distances between census tracts and green zones.
     - **Green Zone Areas**: Larger green spaces have higher scores.
     - **Catchment Area**: A 1000-meter isochrone is used to define the accessible area around each green space.

3. **Spatial Autocorrelation**:
   - **Hot Spot and Cold Spot Analysis**: Spatial autocorrelation analysis is applied to identify clusters of census tracts with high or low green coverage.

4. **Geographically Weighted Regression (GWR)**:
   - **Correlation with Land Surface Temperature**: GWR is used to examine the relationship between green coverage and land surface temperature, helping to understand how vegetation affects urban heat.

## Tools and Technologies

- **Python**: Primary language used for data manipulation and analysis.
- **GeoPandas**: For working with geospatial data.
- **PySAL**: For spatial autocorrelation and hot spot analysis.
- **GWR**: To perform geographically weighted regression.
- **APIs**: To retrieve data from public sources.
- **Spyder IDE**: Used for organizing code in different notebooks for data loading, exploration, and spatial analysis.

## Results

The analysis provides insights into:
- Areas with insufficient green spaces per capita.
- Zones with low accessibility to green zones.
- Hot spots and cold spots of green coverage.
- The relationship between green spaces and land surface temperature.

## Conclusion

The findings can be used by urban planners and policymakers to address inequalities in access to green spaces and to implement strategies to improve environmental conditions in densely populated areas. The study also highlights the importance of green coverage in mitigating urban heat, particularly in regions with high land surface temperatures.
