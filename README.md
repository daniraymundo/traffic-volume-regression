# traffic-volume-regression
A regression model for predicting traffic volume using PyCaret and its built-in traffic dataset.

# Project Background
GMetro is a government traffic operations center that aims to manage daily vehicle congestion in busy areas. Currently, traffic planning relies on static schedules and historical patterns that don’t adjust for dynamic conditions, leading to inefficient road usage.

GMetro wants to build a predictive model that can predict traffic volume based on dynamic factors like weather, holidays, and rush hour patterns. Their goal is to improve resource allocation (e.g., traffic signals, lane closures, public transport dispatching) and reduce traffic congestion.

To be considered effective, the model’s predictions should be within 30% of the actual traffic volume on average. This level of accuracy would give the city enough confidence to act on the model’s forecasts and achieve their goal of smoother, data-informed traffic management.

# Data Structure & Initial Checks

The dataset contains hourly data on traffic volume. It consists of 48,204 rows and 8 columns. A description of each column is as follows:
- **holiday:** a categorical variable that indicates what holiday is celebrated on the date
- **temp:** a numeric variable that shows the average temperature in kelvin.
- **rain_1h:** a numeric variable that shows the amount of rain in mm that occurred in the hour.
- **snow_1h:** a numeric variable that shows the amount of snow in mm that occurred in the hour.
- **clouds_all:** a numeric variable that shows the percentage of cloud cover.
- **weather_main:** a categorical variable that gives a short textual description of the current weather (such as Clear, Clouds, Rain, etc.).
- **rush_hour:** a binary indicator where 1=rush hour, 0=not rush hour
- **traffic_volume:** a numeric variable that shows the hourly reported traffic volume.

### Cleaning and Preprocessing Procedures Performed
- Dropped duplicate rows (244)
- Converted the 'holiday' column to a binary indicator where 1=any holiday, 0=no holiday
- Converted the 'rain_1h' and 'snow_1h' columns to binary indicators where 1=rain/snow was recorded in that hour, 0=no rain/snow recorded
- Re-categorized 'weather_main' to 3 categories instead of 11, encoded ordinally according to severity
- Excluded data where traffic volume <= 1,000
- Enabled PyCaret's default preprocessing
- Removed outliers and highly related features
- Scaled all values of features using minmax scaler (0 to 1 range)
- After preprocessing, the dataset contains 37,029 rows


# Executive Summary

### Overview of Findings
A **Voting Regressor** blending **LightGBM** and **Random Forest** was selected as the best model. Blending these two models combines their strengths, leading to a better general prediction and better performance compared to just using one of the models on its own. It achieved a **MAPE of 29.16%** on unseen data, meeting GMetro's requirement of predictions being **within 30% of actual volumes** on average.


# Model Design
5% of the data was sampled as unseen data. Several different models were compared using PyCaret's compare_models function, evaluating them on 6 metrics: MAE, MSE, RMSE, R2, RMSLE, and MAPE. LightGBM had the best overall performance across multiple evaluation metrics (MSE, RMSE, R2, and RMSLE), and the Random Forest Regressor achieved the lowest MAPE. The models were then tuned using PyCaret's tune_model function, optimizing for MAPE. The tuned LightGBM model was used but the Random Forest model performed better before tuning so the base model was used. These two models were then blended using PyCaret's blend_models function, optimizing for MAPE. The blended model was used to retrain on the entire data and predict on the unseen data.


# Insights Deep Dive
### Model Performance:

* The model is able to explain 48% of traffic volume variation on unseen data. This indicates that the model is performing moderately well considering that traffic is influenced by many unpredictable factors.
  
* The typical prediction error on unseen data is about 868 vehicles per hour. The model tends to underestimate higher traffic volumes and overestimate lower traffic volumes. 

* Despite model selection and tuning, the best model performed significantly better on the train set than on the test set, suggesting overfitting. Further improvements can be made on the data quality to improve model performance.
  
### Feature Importance:

* Temperature and percentage of cloud cover were consistently the two most important features affecting the prediction of traffic volume across the two models. The rest of the features' contributions were minimal. This suggests the absence of more meaningful features than can influence and better predict traffic volume.

# Recommendations:

Based on the insights and findings above, I would recommend GMetro to consider the following: 

* Traffic volume is influenced by a lot of factors and the dataset is very limited. It lacks meaningful features that could better explain the variations. Collect additional data to provide the model with a broader, more diverse foundation for learning. This could include exact date and timestamps and other factors such as road closures, accidents, and special events.
  
* Since the model achieved the target success metric, its predictions are reliable enough for dynamic decisions such as adjusting traffic signals, deploying congestion measures, or planning transit services. It will enable GMetro to move from static traffic schedules to responsive, real-time strategies. Deploy the model in the pilot phase, keeping in mind that it works best for predicting volumes during high-traffic situations but it is not optimized for predictions on very low traffic periods.
  

# Assumptions and Caveats:

Throughout the analysis, multiple assumptions were made to manage challenges with the data. These assumptions and caveats are noted below:

* Null values in 'holiday' (over 99%) were assumed to mean that it was not a holiday and therefore encoded as 0 when converting to binary.
  
* Traffic volume predictions are unreliable when the traffic is very low. This is because the percentage error becomes extremely large even if the absolute error is small.
To ensure that the model meets the goal of being within 30% error for practical, decision-making situations — like rush hours or busy periods — the model was focused on **predicting traffic volumes above 1,000 vehicles per hour**. This makes the model highly reliable during times when traffic management actions (like adjusting signals or rerouting traffic) are actually needed.
