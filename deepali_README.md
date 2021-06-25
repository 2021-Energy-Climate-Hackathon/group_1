I will add comments as I go.

### pre-processor.ipynb :  script to read, filter and average ERA5 data over the UK and convert it to a time series. I have extened the script to do this over multiple years and include week numbers and year as input features. The output variable can be modified to any interesting quantity. The script will generate an output dataframe with feature vectors = hourly values. So we will train our model to learn from daily averages to predict 24 features corresponding to each hour. The script returns an input dataframe (saved as _set1.dat or _set2.dat) and an output dataframe (_wind10.dat or _t2m.dat). Please refer to the branch Luca for a better pre processing script that generates one csv files with all variables.

### feature-selection.ipynb : This script has some useful plotting commands to visualize the correlations between the inputs and the outputs.

### wind10_*.ipynb are tests on wind downscaling with different machine learning approaches and various combinations of input features.

### t2m_*.ipynb are similar tests on temperature

### ssrd_*.ipynb on shortwave solar irradiation

### normalised_seasonal_iris_RMSE_*py and Simple_polyfit3_wind.py 

### the csv and dat files are input files for different py scripts



