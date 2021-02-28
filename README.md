## Bike Sharing Demand-Prediction (Kaggle Competition)
https://www.kaggle.com/c/bike-sharing-demand/overview

---


This project uses Kaggle data to predict Bike Sharing Demand on an hourly basis.   

My personal motivation for this purpose was to create a workflow that allows for an optimal 

For this purpose, I wrote a function that allows adjust all relevant parameters related to variable selection, training and scoring through a "dashboard", making it easy to quickly try out different specifications of the set of explanatory variables/model parameters. 

<br/>

![](https://github.com/NaderH84/Bike_Demand_Prediction-Kaggle-/blob/main/control_panel.png)

<br/>

<br/>

A <strong> click on the image below </strong> opens an interactive Plotly Bubble Plot that provides country-level information on the average cases and deaths from the start of the pandemic up to the first days of Janaury 2021:

<br/>

[![name](plotly_bubble.gif)](https://raw.githack.com/spicedacademy/tensor-tarragon-student-code/nader/week2/interactive_cases.html?token=AGFS44KJGTQMHXBC5GND6ODABP5MQ)

<br/>
<br/>

### Country-Level information on total cases and deaths [up to January 2020]

A <strong> click on the image below </strong> opens an interactive Folium Map that provides country-level information on the total cases and deaths from the start of the pandemic up to the first days of Janaury 2021: 

[![name](World_Map.gif)](https://raw.githack.com/spicedacademy/tensor-tarragon-student-code/nader/week2/world_map.html?token=AGFS44NMP25MF7XVC2TDL23AA7PSG)

<br/>
<br/>

### Country-Region-Level information on new cases in preceding 14 day window as of January 3, 2021

A <strong> click on the image below </strong> opens an interactive Folium Map that provides country-region-level information on the new cases per 100,000 inhabitants during the past 14 days as of January 3, 2021:

<br/>

[![name](European_subnational.gif)](https://raw.githack.com/spicedacademy/tensor-tarragon-student-code/nader/week2/europe_subnational_map.html?token=AGFS44J27SIQHSTYCJJNYKTABP5HI)
<br/>
<br/>
<br/>

---

The underlying data is available at https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide and https://www.ecdc.europa.eu/en/publications-data/subnational-14-day-notification-rate-covid-19


The underlying code of all plots presented above is partly based on modified code snippets from Spiced Academy. Moreover, for the generation of the above Folium Maps, the code provided on this website was used as a starting point: https://towardsdatascience.com/using-python-to-create-a-world-map-from-a-list-of-country-names-cd7480d03b10


### Bike 

https://www.kaggle.com/c/bike-sharing-demand
