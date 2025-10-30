#!/usr/bin/env python
# coding: utf-8

# In[1]:


#package load
import statsmodels.api as sm
from itertools import combinations
from statsmodels.compat import lzip
import statsmodels.stats.api as sms
import itertools
from IPython.display import Markdown, HTML, display
import xgboost as xg
from sklearn.metrics import mean_squared_error as MSE
from matplotlib import pyplot as plt
import seaborn as sns
from ATS.Rates.features_selection.boruta import Boruta
import tabulate
import statsmodels.formula.api as smf
import copy
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2 
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale 
from sklearn import linear_model
from sklearn.metrics import r2_score
from statsmodels.stats.diagnostic import breaks_cusumolsresid
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import rankdata
from scipy.stats import boxcox
import itertools
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import shapiro
from scipy.stats import anderson


# In[2]:


#lag of the desk
link='Credit_lag1'

#file load containing CCAR2024, CCAR2025 scenarios & MEVs
dfMEV = pd.read_excel("H:\My_Received_Files\Credit VA combination with MEVs.xlsx", sheet_name= 'MEVs')
dfMEV["TSY_3M10Y_SPREAD"]=dfMEV.apply(lambda x: x['TSY_3M_RT_EOP'] - x['TSY_10Y_RT_EOP'], axis=1)
dfMEV["TSY_2Y10Y_SPREAD"]=dfMEV.apply(lambda x: x['TSY_2Y_RT_EOP'] - x['TSY_10Y_RT_EOP'], axis=1)
dfVA = pd.read_excel("H:\My_Received_Files\Credit VA combination with MEVs.xlsx", sheet_name= 'VA')

FEDSA_2025=pd.read_excel("H:\My_Received_Files\CCAR2025_multiplesc_Vertical_core_M.xlsx", sheet_name= 'forecast-M-FEDSEVADV')
FEDSA_2025["TSY_3M10Y_SPREAD"]=FEDSA_2025.apply(lambda x: x['TSY_3M_RT_EOP'] - x['TSY_10Y_RT_EOP'], axis=1)
FEDSA_2025["TSY_2Y10Y_SPREAD"]=FEDSA_2025.apply(lambda x: x['TSY_2Y_RT_EOP'] - x['TSY_10Y_RT_EOP'], axis=1)

JPMSA_2025=pd.read_excel("H:\My_Received_Files\CCAR2025_multiplesc_Vertical_core_M.xlsx", sheet_name= 'forecast-M-JPMSEVADV')
JPMSA_2025["TSY_3M10Y_SPREAD"]=JPMSA_2025.apply(lambda x: x['TSY_3M_RT_EOP'] - x['TSY_10Y_RT_EOP'], axis=1)
JPMSA_2025["TSY_2Y10Y_SPREAD"]=JPMSA_2025.apply(lambda x: x['TSY_2Y_RT_EOP'] - x['TSY_10Y_RT_EOP'], axis=1)

JPMAdv_2025=pd.read_excel("H:\My_Received_Files\CCAR2025_multiplesc_Vertical_core_M.xlsx", sheet_name= 'forecast-M-JPMADV')
JPMAdv_2025["TSY_3M10Y_SPREAD"]=JPMAdv_2025.apply(lambda x: x['TSY_3M_RT_EOP'] - x['TSY_10Y_RT_EOP'], axis=1)
JPMAdv_2025["TSY_2Y10Y_SPREAD"]=JPMAdv_2025.apply(lambda x: x['TSY_2Y_RT_EOP'] - x['TSY_10Y_RT_EOP'], axis=1)

FEDSA_2024=pd.read_excel("H:\My_Received_Files\CCAR2024_multiplesc_Vertical_core_M.xlsx", sheet_name= 'forecast-M-FEDSEVADV')
FEDSA_2024["TSY_3M10Y_SPREAD"]=FEDSA_2024.apply(lambda x: x['TSY_3M_RT_EOP'] - x['TSY_10Y_RT_EOP'], axis=1)
FEDSA_2024["TSY_2Y10Y_SPREAD"]=FEDSA_2024.apply(lambda x: x['TSY_2Y_RT_EOP'] - x['TSY_10Y_RT_EOP'], axis=1)

JPMSA_2024=pd.read_excel("H:\My_Received_Files\CCAR2024_multiplesc_Vertical_core_M.xlsx", sheet_name= 'forecast-M-JPMSEVADV')
JPMSA_2024["TSY_3M10Y_SPREAD"]=JPMSA_2024.apply(lambda x: x['TSY_3M_RT_EOP'] - x['TSY_10Y_RT_EOP'], axis=1)
JPMSA_2024["TSY_2Y10Y_SPREAD"]=JPMSA_2024.apply(lambda x: x['TSY_2Y_RT_EOP'] - x['TSY_10Y_RT_EOP'], axis=1)

JPMAdv_2024=pd.read_excel("H:\My_Received_Files\CCAR2024_multiplesc_Vertical_core_M.xlsx", sheet_name= 'forecast-M-JPMADV')
JPMAdv_2024["TSY_3M10Y_SPREAD"]=JPMAdv_2024.apply(lambda x: x['TSY_3M_RT_EOP'] - x['TSY_10Y_RT_EOP'], axis=1)
JPMAdv_2024["TSY_2Y10Y_SPREAD"]=JPMAdv_2024.apply(lambda x: x['TSY_2Y_RT_EOP'] - x['TSY_10Y_RT_EOP'], axis=1)

#Initializing the desk and MEV names to run further code 
df = pd.read_excel("H:\My_Received_Files\RAQ32023_JPMADV.xlsx", sheet_name= 'type')
df_cat=list(set(df['VA']))
df_cat.sort()
df_cat


# In[3]:


#Select the desk from the abve list, considering 'AMERICAS BOLI' is at 0th position

VA=df_cat[3]
VA_diff1= df[ (df['VA'] == VA) ]['VA_diff']
VA_diff1=list(set(VA_diff1))
VA_diff=VA_diff1[0]




VA_lag1=df[ (df['VA'] == VA) ]['VA_lag']
VA_lag1=list(set(VA_lag1))
VA_lag=VA_lag1[0]


#Loading Headers from inventory name and mapping file 
dfMEVCategory1 = pd.read_excel("H:\My_Received_Files\Inventory and name mapping(Credit).xlsx", sheet_name = 'Usable')
dfMEVCategory = dfMEVCategory1[ (dfMEVCategory1['check'] == 'Y') ]
dfMEVCategory2= dfMEVCategory1[ (dfMEVCategory1['Check2'] == 'Y') ]


given_mev=dfMEVCategory['name'].tolist()

dfMEV.set_index('YYYYMM', inplace = True)
dfVA.set_index('YYYYMM', inplace = True)

dfMEV=dfMEV[given_mev]
FEDSA_2025=FEDSA_2025[given_mev]
JPMSA_2025=JPMSA_2025[given_mev]
JPMAdv_2025=JPMAdv_2025[given_mev]
FEDSA_2024=FEDSA_2024[given_mev]
JPMSA_2024=JPMSA_2024[given_mev]
JPMAdv_2024=JPMAdv_2024[given_mev]

dfVA=dfVA[VA]

#date range for training and test data
trainStartDate = 201601
trainEndDt = 202312
testStartDt = 202312
testEndDt = 202412
dfMEV=dfMEV[(dfMEV.index >= trainStartDate) & (dfMEV.index <= testEndDt)]
dfMEV.dropna(axis=1,inplace=True)

#filtering our dataframe using dates
dfMEV=dfMEV[(dfMEV.index >= trainStartDate) & (dfMEV.index <= testEndDt)]


#dropping any column containing NA in this time period
dfMEV.dropna(axis=1,inplace=True)
FEDSA_2025.dropna(axis=1,inplace=True)
JPMSA_2025.dropna(axis=1,inplace=True)
JPMAdv_2025.dropna(axis=1,inplace=True)
FEDSA_2024.dropna(axis=1,inplace=True)
JPMSA_2024.dropna(axis=1,inplace=True)
JPMAdv_2024.dropna(axis=1,inplace=True)

#intializing training and testing data
x_train = dfMEV[(dfMEV.index >= trainStartDate) & (dfMEV.index <= trainEndDt)]    
x_test = dfMEV[(dfMEV.index >= testStartDt) & (dfMEV.index <= testEndDt)]

y_train = dfVA[(dfVA.index >= trainStartDate) & (dfVA.index <= trainEndDt)]    
y_test = dfVA[(dfVA.index >= testStartDt) & (dfVA.index <= testEndDt)]

#getting initial value for CCAR2024 and CCAR2025
initial_value1=y_test.iloc[-1]
initial_value2=y_train.iloc[-1]
print(initial_value1,initial_value2)


# In[4]:


#created function for lag and difference
def createLag(df, period = 1):
    dfshifted = df.shift(period)
    dfshifted = dfshifted.add_suffix('_lag'+str(period))
    dfNew = pd.concat([df, dfshifted], axis=1)
    return dfNew

def createDiff(df, period = 1):
    for xName in df.columns:
        if ('lag' in xName) == False:
            df[xName + '_diff' + str(period)] = df[xName].diff(period)
    return df


# In[5]:


y_train=pd.DataFrame(y_train)
y_test=pd.DataFrame(y_test)


#create lagged dfs
x_train_lag = createLag(x_train, period = 1)
x_test_lag = createLag(x_test, period = 1)
y_train_lag = createLag(y_train, period = 1)
y_test_lag = createLag(y_test, period = 1)
FEDSA_2025_lag = createLag(FEDSA_2025, period = 1)
JPMSA_2025_lag = createLag(JPMSA_2025, period = 1)
JPMAdv_2025_lag = createLag(JPMAdv_2025, period = 1)
FEDSA_2024_lag = createLag(FEDSA_2024, period = 1)
JPMSA_2024_lag = createLag(JPMSA_2024, period = 1)
JPMAdv_2024_lag = createLag(JPMAdv_2024, period = 1)


#create diff dfs
X_train = createDiff(x_train_lag, period = 1)
X_test = createDiff(x_test_lag, period = 1)
Y_train1 = createDiff(y_train_lag, period = 1)
Y_test1 = createDiff(y_test_lag, period = 1)

FEDSA_2025 = createDiff(FEDSA_2025_lag, period = 1)
JPMSA_2025 = createDiff(JPMSA_2025_lag, period = 1)
JPMAdv_2025 = createDiff(JPMAdv_2025_lag, period = 1)
FEDSA_2024 = createDiff(FEDSA_2024_lag, period = 1)
JPMSA_2024 = createDiff(JPMSA_2024_lag, period = 1)
JPMAdv_2024 = createDiff(JPMAdv_2024_lag, period = 1)


### drop extra columns from dependent variable dataframe
filter_col1 = [col for col in Y_train1 if col.endswith('_diff1')]
Y_train=Y_train1[filter_col1]
Y_test=Y_test1[filter_col1]

Final_data_train = pd.concat([Y_train,X_train], axis=1, join='inner') 
Final_data_train.dropna(inplace=True,axis=0)
Final_data_train.head()
Final_data_test = pd.concat([Y_test,X_test], axis=1, join='inner') 
Final_data_test.dropna(inplace=True,axis=0)

FEDSA_2025.dropna(inplace=True,axis=0)
JPMSA_2025.dropna(inplace=True,axis=0)
JPMAdv_2025.dropna(inplace=True,axis=0)
FEDSA_2024.dropna(inplace=True,axis=0)
JPMSA_2024.dropna(inplace=True,axis=0)
JPMAdv_2024.dropna(inplace=True,axis=0)

drop_col=[VA_diff]
X_train=Final_data_train.drop(drop_col, axis=1)
X_test=Final_data_test.drop(drop_col, axis=1)
Y_train=Final_data_train[VA_diff]
Y_test=Final_data_test[VA_diff]






# In[6]:


y_lag_train=Y_train1[link].dropna(axis=0)
y_lag_test=Y_test1[link].dropna(axis=0)
X_train=pd.concat([X_train,y_lag_train], axis=1, join='inner')
X_test=pd.concat([X_test,y_lag_test], axis=1, join='inner')
print('Shape -->',X_train.shape, X_test.shape)
FEDSA_2025[link]=None
JPMSA_2025[link]=None
JPMAdv_2025[link]=None
FEDSA_2024[link]=None
JPMSA_2024[link]=None
JPMAdv_2024[link]=None


# In[7]:


#VIF function
def calculate_vif(X):
    """Calculate VIF for each feature in the DataFrame X."""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data


#function performing regression and all the test
def best_subset_regression(combo,X_train, Y_train, X_test, Y_test,list1, FEDSA_2025, JPMSA_2025, JPMAdv_2025, FEDSA_2024, JPMSA_2024, JPMAdv_2024,initial_value1,initial_value2,link):
    df_independent = X_train
    df_dependent = pd.DataFrame(Y_train)
    results_list1 = []
    results_list2 = []
    results_list3 = []
    
    # Convert tuple to list
    combo_list = list(list1)
   
    #standerdizing independent variables in training data
    scaler = StandardScaler()
    final_df1_train = scaler.fit_transform(X_train[combo_list])
    final_df1_train = pd.DataFrame(final_df1_train)
    final_df1_train.columns = X_train[combo_list].columns.values
    final_df1_train = final_df1_train.set_index(X_train[combo_list].index)
    
    #getting mean and std from training standerdized data
    means = scaler.mean_
    stds = scaler.scale_

    print("Means:")
    print(means)
    print("\nStandard Deviations:")
    print(stds)
    
    #using train standerdize fit to fit test data and scenario mevs
    final_df1_test = scaler.transform(X_test[combo_list])
    final_df1_test = pd.DataFrame(final_df1_test)
    final_df1_test.columns = X_test[combo_list].columns.values
    final_df1_test = final_df1_test.set_index(X_test[combo_list].index)
    
    FEDSA_2025= scaler.transform(FEDSA_2025[combo_list])
    JPMSA_2025= scaler.transform(JPMSA_2025[combo_list])
    JPMAdv_2025= scaler.transform(JPMAdv_2025[combo_list])
    FEDSA_2024= scaler.transform(FEDSA_2024[combo_list])
    JPMSA_2024= scaler.transform(JPMSA_2024[combo_list])
    JPMAdv_2024= scaler.transform(JPMAdv_2024[combo_list])
    
    FEDSA_2025=pd.DataFrame(FEDSA_2025)
    FEDSA_2025.columns=X_test[combo_list].columns.values

    JPMSA_2025=pd.DataFrame(JPMSA_2025)
    JPMSA_2025.columns=X_test[combo_list].columns.values
    
    JPMAdv_2025=pd.DataFrame(JPMAdv_2025)
    JPMAdv_2025.columns=X_test[combo_list].columns.values
   
    FEDSA_2024=pd.DataFrame(FEDSA_2024)
    FEDSA_2024.columns=X_test[combo_list].columns.values
 
    JPMSA_2024=pd.DataFrame(JPMSA_2024)
    JPMSA_2024.columns=X_test[combo_list].columns.values
    #JPMSA_2024=JPMSA_2024.dropna()
    
    JPMAdv_2024=pd.DataFrame(JPMAdv_2024)
    JPMAdv_2024.columns=X_test[combo_list].columns.values
    #JPMAdv_2024=JPMAdv_2024.dropna()
    
    
    
    X=final_df1_train
    X_test=final_df1_test
    y_test=Y_test
    
    
    #Polynomial feature generation (degree 2, including interaction terms) 
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False) 
    X_poly = poly.fit_transform(X)
    X_poly_test = poly.fit_transform(X_test)

    # Get feature names automatically from the PolynomialFeatures 
    feature_names = poly.get_feature_names(input_features=['X1', 'X2','X3'])

    # Create a DataFrame of features and target 
    df = pd.DataFrame(X_poly, columns=feature_names)
    df1 = pd.DataFrame(X_poly_test, columns=feature_names)

    # Full model (using all features)
    X_full = df 
    X_full = X_full.set_index(X.index)
    terms = X_full
    
    
    
    # Full model (using all features)
    X_full_test = df1 
    X_full_test = X_full_test.set_index(X_test.index)
    
  
    # drop terms
    X_poly1 = X_full.drop(list(combo), axis=1)  
    X_poly_test = X_full_test.drop(list(combo), axis=1)
    
    
    # Ensure indices are aligned
    Y_train = Y_train.reset_index(drop=True)
    X_poly1 = sm.add_constant(X_poly1, has_constant='add').reset_index(drop=True)
    
      
    # Fit polynomial regression model using OLS
    model4 = sm.OLS(Y_train, X_poly1).fit()
    y_pred4 = model4.predict(X_poly1)
    print(Y_train,y_pred4)
    
    # Get robust p-values
    robust_pvalues = model4.get_robustcov_results(cov_type='HC1').pvalues

    # Convert p-values to decimal format
    decimal_pvalues = [round(p, 3) for p in robust_pvalues]
    formatted_pvalues = [f"{p:.3f}" for p in decimal_pvalues]
    
    # Check if all robust p-values are significant, excluding the constant
    significance_level = 0.05
    
    # Exclude the first p-value (constant) by slicing the array
    all_significant = np.all(robust_pvalues[1:] < significance_level)

    # Calculate Adjusted R-squared
    n = len(Y_train)  # Number of observations
    p = X_poly1.shape[1] - 1  # Number of predictors (excluding the intercept)
    adjusted_r_squared = 1 - (1 - model4.rsquared) * (n - 1) / (n - p - 1)
    
    #VIF test
    vif_data = calculate_vif(X_poly1)
    print(vif_data[1:])
    
    # Predictions
    y_train_pred = y_pred4
    y_test_pred = model4.predict(sm.add_constant(X_poly_test, has_constant='add'))
    print(y_test,y_test_pred)
    print(model4.summary())
    test_rsq = r2_score(y_test, y_test_pred)
    residuals = Y_train - y_train_pred
    print(formatted_pvalues,test_rsq)
    
    if all_significant and all(vif_data["VIF"][1:] <= 10): 

        #Create DataFrames for predictions
        y_test_pred_df = pd.DataFrame(y_test_pred, index=y_test.index, columns=['Y_test_pred'])

        if link not in final_df1_test.columns:
            # polyno,ia transformation for scenarios of CCAR 2024 and CCAR 2025
            FEDSA_2025_test = poly.fit_transform(FEDSA_2025)
            JPMSA_2025_test = poly.fit_transform(JPMSA_2025)
            JPMAdv_2025_test = poly.fit_transform(JPMAdv_2025)
            FEDSA_2024_test = poly.fit_transform(FEDSA_2024)
            JPMSA_2024_test = poly.fit_transform(JPMSA_2024)
            JPMAdv_2024_test = poly.fit_transform(JPMAdv_2024)

            FEDSA_2025_test1 =pd.DataFrame(FEDSA_2025_test, columns=feature_names)
            JPMSA_2025_test1 =pd.DataFrame(JPMSA_2025_test, columns=feature_names)
            JPMAdv_2025_test1 =pd.DataFrame(JPMAdv_2025_test, columns=feature_names)
            FEDSA_2024_test1 =pd.DataFrame(FEDSA_2024_test, columns=feature_names)
            JPMSA_2024_test1 =pd.DataFrame(JPMSA_2024_test, columns=feature_names)
            JPMAdv_2024_test1 =pd.DataFrame(JPMAdv_2024_test, columns=feature_names)
                
            FEDSA_2025_test2=FEDSA_2025_test1.drop(list(combo), axis=1)
            JPMSA_2025_test2=JPMSA_2025_test1.drop(list(combo), axis=1)
            JPMAdv_2025_test2=JPMAdv_2025_test1.drop(list(combo), axis=1)
            FEDSA_2024_test2=FEDSA_2024_test1.drop(list(combo), axis=1)
            JPMSA_2024_test2=JPMSA_2024_test1.drop(list(combo), axis=1)
            JPMAdv_2024_test2=JPMAdv_2024_test1.drop(list(combo), axis=1)
            
            # Ensure the prediction data has the same features as the training data(add constant)
            FEDSA_2025_test2 = sm.add_constant(FEDSA_2025_test2, has_constant='add')
            JPMSA_2025_test2 = sm.add_constant(JPMSA_2025_test2, has_constant='add')
            JPMAdv_2025_test2 = sm.add_constant(JPMAdv_2025_test2, has_constant='add')
            FEDSA_2024_test2 = sm.add_constant(FEDSA_2024_test2, has_constant='add')
            JPMSA_2024_test2 = sm.add_constant(JPMSA_2024_test2, has_constant='add')
            JPMAdv_2024_test2 = sm.add_constant(JPMAdv_2024_test2, has_constant='add')
            
            # Make predictions
            FEDSA_2025_predict = model4.predict(FEDSA_2025_test2)
            JPMSA_2025_predict = model4.predict(JPMSA_2025_test2)
            JPMAdv_2025_predict = model4.predict(JPMAdv_2025_test2)
            FEDSA_2024_predict = model4.predict(FEDSA_2024_test2)
            JPMSA_2024_predict = model4.predict(JPMSA_2024_test2)
            JPMAdv_2024_predict = model4.predict(JPMAdv_2024_test2)

            FEDSA_2025_predict=pd.DataFrame(FEDSA_2025_predict)
            JPMSA_2025_predict=pd.DataFrame(JPMSA_2025_predict)
            JPMAdv_2025_predict=pd.DataFrame(JPMAdv_2025_predict)
            FEDSA_2024_predict=pd.DataFrame(FEDSA_2024_predict)
            JPMSA_2024_predict=pd.DataFrame(JPMSA_2024_predict)
            JPMAdv_2024_predict=pd.DataFrame(JPMAdv_2024_predict)

            FEDSA_2025_predict.columns = ['FEDSA_2025_predict']
            JPMSA_2025_predict.columns = ['JPMSA_2025_predict']
            JPMAdv_2025_predict.columns = ['JPMAdv_2025_predict']
            FEDSA_2024_predict.columns = ['FEDSA_2024_predict']
            JPMSA_2024_predict.columns = ['JPMSA_2024_predict']
            JPMAdv_2024_predict.columns = ['JPMAdv_2024_predict']
 
            combined_df = pd.concat([FEDSA_2025_predict, JPMSA_2025_predict, JPMAdv_2025_predict, FEDSA_2024_predict, JPMSA_2024_predict, JPMAdv_2024_predict], axis=1)
            print(combined_df)
            
        else:


            # Initialize a list to store predictions
            FEDSA_2025_predict1 = []
            JPMSA_2025_predict1 = []
            JPMAdv_2025_predict1 = []
            FEDSA_2024_predict1 = []
            JPMSA_2024_predict1 = []
            JPMAdv_2024_predict1 = []

            Credit_lag1_index = combo_list.index(link)
            mean_Credit_lag = scaler.mean_[Credit_lag1_index]
            scale_Credit_lag = scaler.scale_[Credit_lag1_index]
            # Use the standardized initial prediction directly




            # Set the initial standardized value for the 0th row's Credit_lag1
            for df in [FEDSA_2025, JPMSA_2025, JPMAdv_2025]:
                df.iloc[0, df.columns.get_loc(link)] = (initial_value1- mean_Credit_lag)/scale_Credit_lag
                
            for df in [FEDSA_2024, JPMSA_2024, JPMAdv_2024]:
                df.iloc[0, df.columns.get_loc(link)] = (initial_value2- mean_Credit_lag)/scale_Credit_lag
                
                
            def update_predictions_and_features(df, predictions_list, drop_columns, str1, initial_value):
                for i, (index, row) in enumerate(df.iterrows()):
                    # Update polynomial features for the current row
                    features_df = df.iloc[i].values.reshape(1, -1)
                    if (np.isnan(features_df).any() or
                        np.isinf(features_df).any() or
                        (np.abs(features_df) > 1e10).any()):
                        continue
                        
                        
                    
                    # features_df1 = features_df.drop(columns='stress_event', errors='ignore')
                    features_poly = poly.transform(features_df)

                    # Get feature names
                    feature_names = poly.get_feature_names(input_features=['X1', 'X2','X3'])

                    # Create a DataFrame with polynomial features
                    features_poly_df = pd.DataFrame(features_poly, columns=feature_names)

                    # Drop specified columns
                    features_poly_df_dropped = features_poly_df.drop(columns=list(drop_columns), errors='ignore')
                    
                    # Add stress_event back to the DataFrame
                    #features_poly_df_dropped['stress_event'] = stress_event_value
                    
                    # Add intercept term
                    features_poly_df_dropped_with_const = sm.add_constant(features_poly_df_dropped, has_constant='add')

                    # Make prediction using the updated polynomial features
                    prediction = model4.predict(features_poly_df_dropped_with_const)

                    prediction1 = prediction + initial_value
                    initial_value1 = prediction1

                    # Standardize the prediction
                    prediction_std = (prediction1 - mean_Credit_lag) / scale_Credit_lag
                    predictions_list.append(prediction[0])
                    
                    # Update the Credit_lag1 for the next row with the standardized prediction
                    if i < len(df) - 1:
                        df.iloc[i + 1, df.columns.get_loc(link)] = prediction_std[0]
                        
                return predictions_list
            # Update predictions and features for each dataset
            FEDSA_2025_predict1=update_predictions_and_features(FEDSA_2025, FEDSA_2025_predict1, combo,"fed_sa_25",initial_value1)
            JPMSA_2025_predict1=update_predictions_and_features(JPMSA_2025, JPMSA_2025_predict1, combo,"jpm_sa_25",initial_value1)
            JPMAdv_2025_predict1=update_predictions_and_features(JPMAdv_2025, JPMAdv_2025_predict1, combo,"ja_sa_25",initial_value1)
            FEDSA_2024_predict1=update_predictions_and_features(FEDSA_2024, FEDSA_2024_predict1, combo,"fed_sa_24",initial_value2)
            JPMSA_2024_predict1=update_predictions_and_features(JPMSA_2024, JPMSA_2024_predict1, combo,"jpm_sa_24",initial_value2)
            JPMAdv_2024_predict1=update_predictions_and_features(JPMAdv_2024, JPMAdv_2024_predict1, combo,"ja_sa_24",initial_value2)
            
            # Convert predictions to DataFrames
            FEDSA_2025_predict1 = pd.DataFrame(FEDSA_2025_predict1, columns=['FEDSA_2025_predict'])
            JPMSA_2025_predict1 = pd.DataFrame(JPMSA_2025_predict1, columns=['JPMSA_2025_predict'])
            JPMAdv_2025_predict1 = pd.DataFrame(JPMAdv_2025_predict1, columns=['JPMAdv_2025_predict'])
            FEDSA_2024_predict1 = pd.DataFrame(FEDSA_2024_predict1, columns=['FEDSA_2024_predict'])
            JPMSA_2024_predict1 = pd.DataFrame(JPMSA_2024_predict1, columns=['JPMSA_2024_predict'])
            JPMAdv_2024_predict1 = pd.DataFrame(JPMAdv_2024_predict1, columns=['JPMAdv_2024_predict'])
            
            
            combined_df = pd.concat([FEDSA_2025_predict1, JPMSA_2025_predict1, JPMAdv_2025_predict1, FEDSA_2024_predict1, JPMSA_2024_predict1, JPMAdv_2024_predict1], axis=1)
            print(combined_df)
            
    return y_pred4


# In[8]:


output_directory = "output_results"


# for columns in column_combinations:
list1 = ['BBB_CORP_SP_RT_EOP_diff1', 'CDX_IG_5Y_3M_OPT_VOL_AVG_diff1', 'CPI_INFL_INX_LVLS_AM_diff1']  

#dropped variable from 2 degree polynomial equation
combo= ['X3', 'X1^2', 'X1 X2', 'X1 X3', 'X2^2']

# Run the model with the subset of columns
tt = best_subset_regression(combo,X_train, Y_train, X_test, Y_test, list1, FEDSA_2025, JPMSA_2025, JPMAdv_2025, FEDSA_2024, JPMSA_2024, JPMAdv_2024,initial_value1,initial_value2,link)


# In[ ]:




