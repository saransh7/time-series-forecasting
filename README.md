# time-series-forecasting
An implementation of generalized time series forecasting models

## Inputs
- List
- DataFrame with one column as Date (missing date imputation : 0)
- Prediction Interval

## Quad Classification
Before anything else, time series is classified based on [ADI](https://frepple.com/blog/demand-classification/) and [CV2](https://frepple.com/blog/demand-classification/). This helps in applying relevant models to the category of the time series<br/>
#### Demand Categorization
<img src="archives/forecastability-demand-patterns.png"
     alt="Demand Categorization"
     style="float: center; margin-right: 10px;" 
     width="600" height="310"/>

#### Threshold values
`ADI: 1.33`, `CV2: 0.49`

