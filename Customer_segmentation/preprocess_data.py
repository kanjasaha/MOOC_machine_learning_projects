import numpy as np
import pandas as pd
from sklearn import preprocessing

def check_null(data):

    if (raw_data.isnull().values.any()):
            message=raw_data.isnull().sum()
    else:
            message='No null value in the dataframe'
    return message

def normalize_data(data):
    #normalize all the columns(features) so that all the values in the column lie between 0 and 1
    #this way each features will get equal preference regardless of their actual range
    n_data=pd.DataFrame()
    n_data = pd.DataFrame(preprocessing.normalize(data),columns=data.columns)
    return n_data

#def remove_outliers_by_value(data, column,value):
#    good_data = data.loc[~data.[column].isin(value)]
#    return good_data

def remove_outliers(data, drop_outlier):
    
# For each feature find the data points with extreme high or low values
    log_data=data
    x=[]
    for feature in log_data.keys():
    
        # TODO: Calculate Q1 (25th percentile of the data) for the given feature
        Q1 = np.percentile(log_data[feature],25)

        # TODO: Calculate Q3 (75th percentile of the data) for the given feature
        Q3 = np.percentile(log_data[feature],75)

        # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
        step = 1.5*(Q3-Q1)
        y= log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
        y1=y.index.values
        x.append(y1)
        # Display the outliers
        #outliercount=y.shape[0]
        #print ("'{} Data points considered outliers for the feature '{}':".format(outliercount,feature))

    
        # OPTIONAL: Select the indices for data points you wish to remove
        # Here I go through the lists and extract the index value that is repeated in more than one list.
        seen = set()
        repeated = set()
    for l in x:
        for i in set(l):
            if i in seen:
              repeated.add(i)
            else:
              seen.add(i)

    outliers =list(repeated)
    outlier_count=len(outliers)
    total_count=len(log_data)
       
    percent_outliers=(float(outlier_count)*100)/(float(total_count))
    #display(percent_outliers)
    delete_status = "Outlier not dropped from dataset"
    
    if drop_outlier is True:
        # Remove the outliers, if any were specified
        good_data = data.loc[~data.index.isin(outliers)]
        delete_status = "Outlier Dropped from dataset"
        data=good_data
        
    message =("{} ({:2.2f}%) data points considered outliers from the dataset of {}. {}.".format(outlier_count,percent_outliers,total_count,delete_status))   
    return data,outliers , message
