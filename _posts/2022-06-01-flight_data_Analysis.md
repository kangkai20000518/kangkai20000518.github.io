---
layout: post
title: Flight Dealy Analysis
---

## §1. Introduction
The aim of this project is to use the data set provided by the U.S. Department of Transportation
(DOT) Bureau of Transportation Statistics to analyze factors to predict the number of Total flight and punctuality rate for each day. I
want to use the LSTM(Long Short-Term Memory) to help me get the result. The ultimate goal of the
project is help people travel more efficiently and know the relationship between flight and weather.


```python
from plotly import express as px
from keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, LSTM
import numpy as np
import pandas as pd
import plotly.io as pio
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import sqlite3#sqlite3 python module that lets us interact wtih squile
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import re
import time
import datetime
```

## §2. Data preparation

### 1）Flight Data


```python
Flight2016=pd.read_csv("2016.csv")
Flight2015=pd.read_csv("2015.csv")
Flight2017=pd.read_csv("2017.csv")
Flight2018=pd.read_csv("2018.csv")
Flight2014=pd.read_csv("2014.csv")
Flight2013=pd.read_csv("2013.csv")
Flight2012=pd.read_csv("2012.csv")
```

Because LSTM(Long Short-Term Memory) is one of the deep learning techniques, as much data as possible will help me get more accurate results, so I downloaded all the original flight data from 2012-2018, the following function will perform the most basic data The cleaning work is to facilitate the subsequent splicing of other datasets to form a complete structure.

From the raw data, we found that there are a total of 28 columns in the flight dataset. If the unit of our prediction is every flight instead of every day, I will select predictors through correlation plot or heat map, but in this project, the object we explore is the unit of day, so after data cleaning, Most columns will be removed, or converted to other varibles.


```python
Flight2016.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FL_DATE</th>
      <th>OP_CARRIER</th>
      <th>OP_CARRIER_FL_NUM</th>
      <th>ORIGIN</th>
      <th>DEST</th>
      <th>CRS_DEP_TIME</th>
      <th>DEP_TIME</th>
      <th>DEP_DELAY</th>
      <th>TAXI_OUT</th>
      <th>WHEELS_OFF</th>
      <th>...</th>
      <th>CRS_ELAPSED_TIME</th>
      <th>ACTUAL_ELAPSED_TIME</th>
      <th>AIR_TIME</th>
      <th>DISTANCE</th>
      <th>CARRIER_DELAY</th>
      <th>WEATHER_DELAY</th>
      <th>NAS_DELAY</th>
      <th>SECURITY_DELAY</th>
      <th>LATE_AIRCRAFT_DELAY</th>
      <th>Unnamed: 27</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-01-01</td>
      <td>DL</td>
      <td>1248</td>
      <td>DTW</td>
      <td>LAX</td>
      <td>1935</td>
      <td>1935.0</td>
      <td>0.0</td>
      <td>23.0</td>
      <td>1958.0</td>
      <td>...</td>
      <td>309.0</td>
      <td>285.0</td>
      <td>249.0</td>
      <td>1979.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-01-01</td>
      <td>DL</td>
      <td>1251</td>
      <td>ATL</td>
      <td>GRR</td>
      <td>2125</td>
      <td>2130.0</td>
      <td>5.0</td>
      <td>13.0</td>
      <td>2143.0</td>
      <td>...</td>
      <td>116.0</td>
      <td>109.0</td>
      <td>92.0</td>
      <td>640.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-01-01</td>
      <td>DL</td>
      <td>1254</td>
      <td>LAX</td>
      <td>ATL</td>
      <td>2255</td>
      <td>2256.0</td>
      <td>1.0</td>
      <td>19.0</td>
      <td>2315.0</td>
      <td>...</td>
      <td>245.0</td>
      <td>231.0</td>
      <td>207.0</td>
      <td>1947.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-01-01</td>
      <td>DL</td>
      <td>1255</td>
      <td>SLC</td>
      <td>ATL</td>
      <td>1656</td>
      <td>1700.0</td>
      <td>4.0</td>
      <td>12.0</td>
      <td>1712.0</td>
      <td>...</td>
      <td>213.0</td>
      <td>193.0</td>
      <td>173.0</td>
      <td>1590.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-01-01</td>
      <td>DL</td>
      <td>1256</td>
      <td>BZN</td>
      <td>MSP</td>
      <td>900</td>
      <td>1012.0</td>
      <td>72.0</td>
      <td>63.0</td>
      <td>1115.0</td>
      <td>...</td>
      <td>136.0</td>
      <td>188.0</td>
      <td>121.0</td>
      <td>874.0</td>
      <td>72.0</td>
      <td>0.0</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
conn=sqlite3.connect("project.db")#create data base called project
def flight_data_each_year(Flight):
    Flight.fillna(0,inplace=True)#because NA means there is no delay so we can fill it with 0
    LAX=Flight[Flight["ORIGIN"]=="LAX"]#Because there is too much data for each year of takeoff, hence we will study the planes taking off from LAX
    LAX["delay"]=(LAX[["CARRIER_DELAY","WEATHER_DELAY","NAS_DELAY","SECURITY_DELAY","LATE_AIRCRAFT_DELAY"]] != 0).astype(int).sum(axis=1)
    #There are many reasons for flight delay, and we only care whether a flight is delayed, so we need to add all the reasons, 
    #if it is greater than or equal to 1which means it is delayed
    LAX["delay"][LAX.delay>1]=1#Turn multiple reasons into 1
    cols=["FL_DATE","delay"]#
    LAX=LAX[cols]
    LAX.to_sql("LAX", conn, if_exists = "replace", index = False)#put LAX.csv to the data base
    #Call sql to calculate the total number of flights per day and the total number of delayed flights 
    #(because when delay==0, it means there is no delay, so you only need to add all the delays to know the total number of delays)
    cmd = """
       SELECT *, count() AS Total_flight_per_day, SUM(delay) AS Total_Delay_per_day
       FROM LAX
       GROUP BY FL_DATE
       
         """
    LAX_flight = pd.read_sql_query(cmd, conn)
    LAX_flight["punctuality rate"]=1-LAX_flight["Total_Delay_per_day"]/LAX_flight["Total_flight_per_day"]#计算每一天的准点率
    LAX_flight["week"] = ''#Create a new column named week
    for i in range(len(LAX_flight["FL_DATE"])):
        #Call the datetime package to standardize the time to get the day of the week for each day
        LAX_flight["week"][i]=datetime.datetime.strptime(LAX_flight["FL_DATE"].iloc[i],"%Y-%m-%d").isoweekday()
        #Because the output result only corresponds to the number of the week, so we need to convert it into English text
        LAX_flight["week"]=LAX_flight["week"].replace({5:'Friday',
                       1:'Monday',
                       2:'Tuesday',
                       3:'Wednesday',
                       4:'Thursday',
                       6:'Saturday',
                       7:'Sunday'
                                    })
    #Because we need to use a neural network in the end, we need to use OneHotEncode for categorical variables
    encoder = OneHotEncoder(handle_unknown='ignore')
    #perform one-hot encoding on 'week' column 
    encoder_df = pd.DataFrame(encoder.fit_transform(LAX_flight[['week']]).toarray())                         
    # #merge one-hot encoded columns back with original DataFrame
    LAX_flight = LAX_flight.join(encoder_df)
    # #view final df
    #LAX_flight.drop(labels=['week'], axis=1, inplace=True)
    colNameDict = {0:'Friday',
                   1:'Monday',
                   2:'Saturday',
                   3:'Sunday',
                   4:'Thursday',
                   5:'Tuesday',
                   6:'Wednesday',
                  }                  
    LAX_flight.rename(columns = colNameDict,inplace=True)
    LAX_flight.drop(labels=['Total_Delay_per_day','delay'],axis=1, inplace=True)
    return LAX_flight
```


```python
LAX_flight2016=flight_data_each_year(Flight2016)
LAX_flight2015=flight_data_each_year(Flight2015)
LAX_flight2017=flight_data_each_year(Flight2017)
LAX_flight2018=flight_data_each_year(Flight2018)
LAX_flight2014=flight_data_each_year(Flight2014)
LAX_flight2013=flight_data_each_year(Flight2013)
LAX_flight2012=flight_data_each_year(Flight2012)
```


```python
LAX_flight=pd.concat([LAX_flight2015,LAX_flight2016,LAX_flight2017,LAX_flight2018])
#Because the amount of data is too large, it is necessary to group the data sets to combine
LAX_flight=pd.concat([LAX_flight,LAX_flight2012,LAX_flight2013,LAX_flight2014])
```


```python
LAX_flight
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FL_DATE</th>
      <th>Total_flight_per_day</th>
      <th>punctuality rate</th>
      <th>week</th>
      <th>Friday</th>
      <th>Monday</th>
      <th>Saturday</th>
      <th>Sunday</th>
      <th>Thursday</th>
      <th>Tuesday</th>
      <th>Wednesday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-01-01</td>
      <td>556</td>
      <td>0.766187</td>
      <td>Thursday</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-01-02</td>
      <td>640</td>
      <td>0.656250</td>
      <td>Friday</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-01-03</td>
      <td>575</td>
      <td>0.526957</td>
      <td>Saturday</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-01-04</td>
      <td>615</td>
      <td>0.536585</td>
      <td>Sunday</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-01-05</td>
      <td>622</td>
      <td>0.667203</td>
      <td>Monday</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>360</th>
      <td>2014-12-27</td>
      <td>564</td>
      <td>0.734043</td>
      <td>Saturday</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>361</th>
      <td>2014-12-28</td>
      <td>604</td>
      <td>0.743377</td>
      <td>Sunday</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>362</th>
      <td>2014-12-29</td>
      <td>627</td>
      <td>0.733652</td>
      <td>Monday</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>363</th>
      <td>2014-12-30</td>
      <td>614</td>
      <td>0.547231</td>
      <td>Tuesday</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>364</th>
      <td>2014-12-31</td>
      <td>493</td>
      <td>0.748479</td>
      <td>Wednesday</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>2557 rows × 11 columns</p>
</div>



We used One-Hot Encoding instead of our commonly used label Encoding in the above processing of the week. Because the premise of using label Encoding is that the variable should behave to some extent, for example, easy is 1, medium is 2, difficult is 3. But in our case the day of the week does not represent such an attribute so I used One- Hot Encoding. In fact, One-Hot Encoding is very easy to understand. Each corresponding value is only 0 and 1. 0 means it does not have the property, and 1 means it has the property. For example, on 2015-01-01, the value corresponding to Thursday is 1, which means that 2015-01-01 is Thursday.

### 2) visulazation


```python
fig = px.line(data_frame = LAX_flight2015, # data that needs to be plotted
                 x = "FL_DATE", # column name for x-axis
                 y = "Total_flight_per_day",
                 markers=True,
                 color = "week", # column name for color coding
                 width = 1000,
                 height = 500)

# reduce whitespace
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# show the plot
fig.show()
```

{% include flight1.html %}

```python
fig = px.line(data_frame = LAX_flight2015, # data that needs to be plotted
                 x = "FL_DATE", # column name for x-axis
                 y = "punctuality rate",
                 markers=True,
                 color = "week", # column name for color coding
                 width = 1000,
                 height = 500)

# reduce whitespace
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# show the plot
fig.show()
pio.write_html(fig, file='flight2.html', auto_open=True)
```
{% include flight2.html %}

According to the image, we can conclude that the overall flight punctuality rate tends to be a stable sequence, that is, a sequence with no obvious upward or downward trend, and the average punctuality rate is about 70%-90%. The punctuality rate is inversely proportional, which is also in line with our intuition that the punctuality rate of the aircraft is not high when the shipping pressure is high. When we look at the data corresponding to the number of days, we find that the weather should be a very important factor. Therefore, the next part of the code is mainly to deal with the weather conditions in the Los Angeles area from 2012 to 2018.

### 3) Weather Data


```python
weather = pd.read_csv("weather2012_2018.csv") 
```


```python
weather.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>local_time LA (airport)</th>
      <th>Ff</th>
      <th>N</th>
      <th>VV</th>
      <th>Td</th>
      <th>RRR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>31.12.2018 22:00</td>
      <td>2.0</td>
      <td>20-0%.</td>
      <td>16</td>
      <td>-12.2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>31.12.2018 16:00</td>
      <td>4.0</td>
      <td>50%.</td>
      <td>16</td>
      <td>6.7</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>31.12.2018 10:00</td>
      <td>2.0</td>
      <td>70 -80%.</td>
      <td>16</td>
      <td>7.8</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>31.12.2018 04:00</td>
      <td>2.0</td>
      <td>50%.</td>
      <td>16</td>
      <td>6.1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>30.12.2018 22:00</td>
      <td>3.0</td>
      <td>20-0%.</td>
      <td>16</td>
      <td>8.3</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Before processing, let's take a look at the raw data. We found that the monitoring station will measure 4 times a day, the first time is 22:00, the second time is 16:00, the third time is 10:10, and the fourth time is 22:00. But don't forget that the unit of our final data is each day, so the time doesn't matter, so in the function below, we will deal with the time. And the varible name of the original data is not very clear, so we have to rename the variable name to help everyone better understand the corresponding real meaning.


```python
def weather_data_each_year(Weather):
    Weather.fillna(0,inplace=True)
    Weather['local_time LA (airport)'] = pd.to_datetime(Weather['local_time LA (airport)'])#normalize time
    Weather["Day_time"]=Weather["local_time LA (airport)"].apply(lambda x:str(x)[0:10])#only need day don't care about hours and min
    cols=["Day_time",'local_time LA (airport)',"Ff","N","RRR","VV"]
    Weather=Weather[cols]
    Weather.columns = ["Day_time",'Observation_time', 'wind_speed',"cloud", 'precipitation', 'visibility']#rename
    Weather["precipitation"]=Weather["precipitation"].replace("Signs of precipitation", 0)
    Weather["cloud"]=Weather["cloud"].replace({'100%.':1,#Convert percentages to corresponding decimals for subsequent calculations
                          '70 -80%.':0.75,
                          '50%.':0.5,
                          0:0,
                         'The sky is not visible due to fog and/or other meteorological phenomena.':0,
                         '20-0%.':0.1,
                          'Cloudless':0
                                   })
    Weather.to_sql("Weather", conn, if_exists = "replace", index = False)#put Weather.csv to the data base"\
    #cmd will calculate average wind speed, average visibility, total precipitation per day
    cmd = """
   
       SELECT Day_time,AVG(wind_speed) AS wind_speed_perday,AVG(visibility) AS visibility_perday, AVG(cloud) AS cloud_perday ,SUM(precipitation) AS precipitation_perday
       FROM Weather
       GROUP BY Day_time
       
         """
    LAX_weather= pd.read_sql_query(cmd, conn)
    #Convert real cloud cover according to real weather
    for i in range(len(LAX_weather["cloud_perday"])):
        if(LAX_weather["cloud_perday"][i]==0):
            LAX_weather["cloud_perday"][i]='cloudless'
        elif(LAX_weather["cloud_perday"][i]<0.3):
            LAX_weather["cloud_perday"][i]='less cloudy'
        elif(LAX_weather["cloud_perday"][i]<0.7):
            LAX_weather["cloud_perday"][i]='cloudy'
        else:
            LAX_weather["cloud_perday"][i]="overcast"  
    #Convert precipitation to heavy, medium and light rain
    for i in range(len(LAX_weather["precipitation_perday"])):
        if(LAX_weather["precipitation_perday"][i]==0):
            LAX_weather["precipitation_perday"][i]="no_rain"
        elif(LAX_weather["precipitation_perday"][i]<10):
            LAX_weather["precipitation_perday"][i]="light_rain"
        elif(LAX_weather["precipitation_perday"][i]<25):
            LAX_weather["precipitation_perday"][i]="moderate_rain"
        else:
            LAX_weather["precipitation_perday"][i]="heavy_rain"   
    #creating instance of one-hot-encoder
    encoder = OneHotEncoder(handle_unknown='ignore')

    #perform one-hot encoding on ['cloud_perday','precipitation_perday'] column 
    encoder_df = pd.DataFrame(encoder.fit_transform(LAX_weather[['cloud_perday','precipitation_perday']]).toarray())

    #merge one-hot encoded columns back with original DataFrame
    LAX_weather = LAX_weather.join(encoder_df)

    #view final df
    LAX_weather.drop(labels=['cloud_perday','precipitation_perday'], axis=1, inplace=True)
    colNameDict = {0:'cloudless',
                   1:'cloudy',
                   2:'fog',
                   3:'less cloudy',
                   4:'overcast',
                   5:'heavy_rain',
                   6:'light_rain',
                   7:'moderate_rain',
                   8:'no_rain'
                  }                  
    LAX_weather.rename(columns = colNameDict,inplace=True)
    return LAX_weather
```


```python
LAX_weather=weather_data_each_year(weather)
LAX_weather
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Day_time</th>
      <th>wind_speed_perday</th>
      <th>visibility_perday</th>
      <th>cloudless</th>
      <th>cloudy</th>
      <th>fog</th>
      <th>less cloudy</th>
      <th>overcast</th>
      <th>heavy_rain</th>
      <th>light_rain</th>
      <th>moderate_rain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2012-01-01</td>
      <td>2.00</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2012-01-02</td>
      <td>1.75</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012-01-03</td>
      <td>2.75</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2012-01-04</td>
      <td>5.50</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012-01-05</td>
      <td>3.50</td>
      <td>13.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2552</th>
      <td>2018-12-27</td>
      <td>5.25</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2553</th>
      <td>2018-12-28</td>
      <td>2.50</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2554</th>
      <td>2018-12-29</td>
      <td>2.00</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2555</th>
      <td>2018-12-30</td>
      <td>3.00</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2556</th>
      <td>2018-12-31</td>
      <td>2.50</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>2557 rows × 11 columns</p>
</div>



From the above table, it can be found that I calculated the average wind speed and visibility for each day, and for the calculation of precipitation I used the sum function to sum instead of the average. This is because the precipitation after sum is a more accurate, but the wind speed and visibility cannot be used in the same way, because it does not have any meaning when we do the addition. Finally, I convert the precipitation and cloud conditions into categorical varibles according to the real division criteria. At the same time, perform One-Hot Encoding processing on it.

### 4) Merge datasets


```python
LAX_weather.to_sql("LAX_weather", conn, if_exists = "replace", index = False)#put temps.csv to the data base"
LAX_flight.to_sql("LAX_flight", conn, if_exists = "replace", index = False)#put temps.csv to the data base"
```




    2557




```python
cmd = """
   SELECT *
   FROM LAX_flight
   LEFT JOIN LAX_weather ON LAX_weather.Day_time = LAX_flight.FL_DATE
       
     """
    
data = pd.read_sql_query(cmd, conn)
```


```python
#min max normalization
for column in data[["wind_speed_perday","visibility_perday","Total_flight_per_day"]]:
    data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())    
```

We call SQL again, because the LEFT JOINT function will help us merge data set. The resulting data is the final dataset. However, at the same time we also normalized the numerical varible in the data. And this time we are using the Min-Max method instead of the standard scaler. Because I found that our data is not a normal distribution, so Min-Max will be more suitable for our project.


```python
sorted_df = data.sort_values(by='FL_DATE')
```


```python
sorted_df=sorted_df.drop(labels=["Day_time","week"],axis=1)
```


```python
sorted_df 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FL_DATE</th>
      <th>Total_flight_per_day</th>
      <th>punctuality rate</th>
      <th>Friday</th>
      <th>Monday</th>
      <th>Saturday</th>
      <th>Sunday</th>
      <th>Thursday</th>
      <th>Tuesday</th>
      <th>Wednesday</th>
      <th>wind_speed_perday</th>
      <th>visibility_perday</th>
      <th>cloudless</th>
      <th>cloudy</th>
      <th>fog</th>
      <th>less cloudy</th>
      <th>overcast</th>
      <th>heavy_rain</th>
      <th>light_rain</th>
      <th>moderate_rain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1461</th>
      <td>2012-01-01</td>
      <td>0.585366</td>
      <td>0.778966</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.200</td>
      <td>0.871176</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1462</th>
      <td>2012-01-02</td>
      <td>0.881098</td>
      <td>0.653495</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.175</td>
      <td>0.871176</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1463</th>
      <td>2012-01-03</td>
      <td>0.673780</td>
      <td>0.806780</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.275</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1464</th>
      <td>2012-01-04</td>
      <td>0.637195</td>
      <td>0.787197</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.550</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1465</th>
      <td>2012-01-05</td>
      <td>0.740854</td>
      <td>0.900327</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.350</td>
      <td>0.838969</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>2018-12-27</td>
      <td>0.908537</td>
      <td>0.656672</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.525</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>2018-12-28</td>
      <td>0.896341</td>
      <td>0.707391</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.250</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>2018-12-29</td>
      <td>0.753049</td>
      <td>0.735390</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.200</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1459</th>
      <td>2018-12-30</td>
      <td>0.868902</td>
      <td>0.813456</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.300</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1460</th>
      <td>2018-12-31</td>
      <td>0.615854</td>
      <td>0.861646</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.250</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>2557 rows × 20 columns</p>
</div>



Before dividing the data into training set and test set, we need to reorder the data, because our standard of division is to divide the data into 60% training set, 20% vaildation group and 20% test set in chronological order rather than randomly.

## §3. training，validation，and testing group


```python
# 60% train, 20% validation, 20% test

train_size = int(0.6*len(sorted_df )) 
val_size = int(0.2*len(sorted_df ))

train = sorted_df [:train_size]
val = sorted_df [train_size : train_size + val_size]
test = sorted_df [train_size + val_size:]
```


```python
from sklearn.model_selection import train_test_split
#split into X and y
X_train=train.drop(['Total_flight_per_day','punctuality rate','FL_DATE'],axis=1)
y_train=train[['Total_flight_per_day','punctuality rate']]
#split into X and y
X_val=val.drop(['Total_flight_per_day','punctuality rate','FL_DATE'],axis=1)
y_val=val[['Total_flight_per_day','punctuality rate']]
#split into X and y
X_test=test.drop(['Total_flight_per_day','punctuality rate','FL_DATE'],axis=1)
y_test=test[['Total_flight_per_day','punctuality rate']]

```


```python
def create_sequence(values, time_steps):
    output = []
    for i in range(len(values) - time_steps):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

```


```python
X_train = create_sequence(X_train, 7)
X_test  = create_sequence(X_test, 7)
X_val = create_sequence(X_val, 7)
y_train = y_train[-X_train.shape[0]:]
y_test  = y_test[-X_test.shape[0]:]
y_val = y_test[-X_val.shape[0]:]
```


```python
X_train.shape
```




    (1527, 7, 17)



The above create_sequence function can convert our 2D data into a 3D array because the input of the LSTM is always a 3D array. We have a total of 1533 training data. So far all the preparatory work is over.

## §4. Model Selection

When we got the data ready we decided to use keras' sequential model to construct the training model. For an ordinary neural network, whenever it completes one thing and outputting the result, it will forget the past, and when new variables enter the network, it will only make predictions based on new inputs. However, human thinking is not like this. Instead of throwing everything away and starting from scratch, we apply old and newly acquired information to get new answers. And RNNs work very similar to the way humans think. But RNN is not omnipotent. When the old information and the new input information are too far away, it will lose the ability to learn such distant information. And in the process, we may lose very important information, making predictions inaccurate. The solution is LSTM, a variant of RNN. LSTM does not remember every information, but remembers important information and brings it into the next operation, and automatically discards unimportant factors. The process of implementation is to introduce three important principles, input gate, an output gate and a forget gate. Considering that the weather will have a continuous impact on the alignment rate, the LSTM recurrent neural network is used here to make predictions, using Dense(2) , so that our output model can output two results, the first result is the prediction effect of the number of flights on the second day, and the second result is the prediction result of the punctuality rate on the second day. The picture below is a sample RNN that can help us to understand the concept.

![RNN.png]({{ site.baseurl }}/images/RNN.png)    

```python
from tensorflow.keras.layers import Bidirectional
model = Sequential()
# Adding a LSTM layer
model.add(LSTM(32,return_sequences=False,input_shape=(X_train.shape[1],X_train.shape[-1])))
model.add(Dense(2))
model.add(tf.keras.layers.Activation('sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='AdaDelta',metrics=['accuracy'])
```


```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm (LSTM)                  (None, 32)                6400      
    _________________________________________________________________
    dense (Dense)                (None, 2)                 66        
    _________________________________________________________________
    activation (Activation)      (None, 2)                 0         
    =================================================================
    Total params: 6,466
    Trainable params: 6,466
    Non-trainable params: 0
    _________________________________________________________________
    


```python
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)
history = model.fit(X_train,
                    y_train,
                    validation_data=(X_val, y_val),
                    epochs = 500,
                    verbose = False,
                    batch_size=50, 
                    callbacks=[callback]
)
```


```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','testing'], loc='upper right')
plt.show()

```


![output_45_0.png]({{ site.baseurl }}/images/output_45_0.png)      


It can be seen that our effect is still good! Because the loss of vaildation has been declining.


```python
import matplotlib.pyplot as plt
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "vaildation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```




    <matplotlib.legend.Legend at 0x1f823099e80>





![output_47_1.png]({{ site.baseurl }}/images/output_47_1.png)        



```python
model.evaluate(X_test,y_test)
```

    16/16 [==============================] - 0s 719us/step - loss: 0.5275 - accuracy: 0.7446
    




    [0.5275416970252991, 0.7445544600486755]



The accuracy of the model finally reached 74% because only the weather factor is used, so the accuracy is pretty good. Of course, during model training, we can also adjust its window size, training times, batch size, dropout ratio, etc. When time is sufficient, the model can be adjusted repeatedly to obtain better prediction results.

## §5. Acknowledgment
Many thanks to our dearest Harlin who provided us with this wonderful to let us practice what we’ve learned in real life cases.