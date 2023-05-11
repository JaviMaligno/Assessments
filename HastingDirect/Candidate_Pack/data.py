# %%
import pandas as pd
import numpy as np

df = pd.read_csv("Grad_Scheme_Case_Study_Dataset.csv")
#len(df["Quote_ID"].unique()) == len(df["Quote_ID"])
#np.sum(df.duplicated())


#I can maybe use duplicates to test the results %https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.duplicated.html
duplicated_quotes = df[df.duplicated(keep="first")]


df[df.duplicated(subset = "Quote_ID",keep="first")] #all duplicates in ID are complete duplicates, there are the same amount, so no id with conflicting info
# PLAN
# Explore uniqueness, missing values etc
# Turn DOB into Age, Check if there are too old quote dates, their price may not be relevant anymore
# replace categorical by dummies (dummy_na=True, possibly drop_first=False)
# See if premium and capped premium are ever different and try to explain this as a price taking into account monthly / yearly
# Use VIF to remove highly correlated variables one by one. Use PCA as well to compare. For instance age and driving licence years, maybe just subtract
# For linear regression it is useful to scale data so that the mean of the features is 0 https://stats.stackexchange.com/questions/29781/when-conducting-multiple-regression-when-should-you-center-your-predictor-varia
# Scaling includes target value as well https://stats.stackexchange.com/questions/111467/is-it-necessary-to-scale-the-target-value-in-addition-to-scaling-features-for-re
# For tree based algorithms it is not necessary to scale
# Plot some of the (numerical) variables with respect to the price I have chosen to see if the relation is linear (maybe plot regression line to make it more obvious)
# Finally do a multilinear regression and get the anova table / do it with sklearn splitting the data etc
# If there's time try alternative algorithms. Potentially decision trees/random forest https://www.analyticsvidhya.com/blog/2021/05/5-regression-algorithms-you-should-know-introductory-guide/#:~:text=Linear%20Regression%20is%20an%20ML,the%20other%20given%20independent%20variables. 
# or even a neural network
# Make sure to fill values to use sklearn algorithms, otherwise use XGBoost https://stackoverflow.com/questions/30317119/classifiers-in-scikit-learn-that-handle-nan-null

# %%
unique_quotes = df.drop_duplicates(keep="first",inplace=False, ignore_index=True) #inplace=True if I want to forget about the original dataset #ignore_index=True if I want to reindex sequentially

#all the quotes are from the same year so the prices are relevant
unique_quotes["Quote_Date"].min(), unique_quotes["Quote_Date"].max() 


# %%
from datetime import datetime
from dateutil.relativedelta import relativedelta
# Since all the quotes are from 2020 I am going to use 01/01/2021 to compute age
# Only driver 1 has DOB and other data, it would probably be useful to have driver 2 as well
# There is surely high correlation between driver licence years and age, so I may subtract the first from the latter to obtain less correlated variables
# This is also where linearity is likely to fail as experience drivers are more trustworthy than new drivers but old drivers might be less trustworthy than mature drivers

unique_quotes["Driver1_DOB"] = pd.to_datetime(unique_quotes["Driver1_DOB"]).apply(lambda x: relativedelta(datetime(2021,1,1),x).years)
unique_quotes.rename(columns={"Driver1_DOB":"Driver1_Age"}, inplace=True)
any(unique_quotes["Driver1_Age"].isnull()) #there are no missing values




# %%
# Let's try to make sense of missing values for each column
# In driver licence columns I'm not filling them, Uncertainty is information too, alternatively in driver licence years the average could be used
# for licence type there is no reasonable choice, NaN could even imply other or no driver2, only with more information e.g. nationality we could infer something
any(unique_quotes["Driver1_Licence_Years"].isnull()) #no mising values here
unique_quotes["Driver1_Age"].corr(unique_quotes["Driver1_Licence_Years"]), unique_quotes["Driver1_Age"].corr(unique_quotes["Driver1_Age"]-unique_quotes["Driver1_Licence_Years"]) 
#correlation is lower but not too much lower, from 0.69 to 0.57, so maybe I'll just use VIF to drop columns or PCA to find new combinations

# %%
# for driver2 licence years it may make sense to fill with average for the purpose of regression, but only when we know there is a driver 2
# if dirver2 licence is NaN we should fill with 0 because we are assuming this means not driver2
# However, the effect of not having a second driver may not have to be the same as having a new driver, so it may actually make sense to fill with average to eliminate the effect of the second driver (it would give the average price)
# or maybe don't fill at all because not having a second driver should be treated differently (specially because we don't know the exact a priori effect of driving years)
# I could create another variable that says whether there is a second driver or no instead
# If there is time, try all these different variations
mean_years = unique_quotes["Driver2_Licence_Years"].mean()
#unique_quotes["Driver2_Licence_Years"] = 
def replace_years(row):
    if (pd.isna(row['Driver2_Licence_Type']) and pd.isna(row['Driver2_Licence_Years'])):
        return 0
    elif pd.isna(row['Driver2_Licence_Years']):
        return mean_years 
    else:
        return row['Driver2_Licence_Years']
# alt fill only if the first one is not missing and leave NaN otherwise
unique_quotes.apply(replace_years, axis=1)
unique_quotes["Driver2_Licence_Years"].fillna(mean_years)


#%%
#There seems to be an erroneous value for convictions, I am going to assume that this should be treated as no convictions since there is only yes or no and innocent until proved wrong
# I will replace with 'No' and later this will turned to dummy
unique_quotes[unique_quotes["Driver1_Convictions"]=='-9999']
unique_quotes["Driver1_Convictions"].replace('-9999','No', inplace=True)


unique_quotes["Driver1_Convictions"].unique()


#%%
any(unique_quotes["Driver1_Claims"].isna()) #no missing values, I'll leave it as it is

#%%
unique_quotes["Driver1_Marital_Status"].unique() #there is single and Single, and  married and Married, change all to capitalized
unique_quotes["Driver1_Marital_Status"] = unique_quotes["Driver1_Marital_Status"].apply(lambda x: x.title())

# %%
#no missing values, all numerical
any(unique_quotes["Vehicle_Age"].isnull()), unique_quotes["Vehicle_Age"].dtypes
any(unique_quotes["Vehicle_Value"].isnull()), unique_quotes["Vehicle_Value"].dtypes
any(unique_quotes["Tax"].isnull()), unique_quotes["Vehicle_Value"].dtypes #how max vechicle tax it paus, it is around 15% of the vehicle value, probably not relevant. It is also based on age of vehicle and emissions https://www.theaa.com/driving-advice/driving-costs/car-tax-explained#need-tax so it is certainly redundant
unique_quotes["Tax"].corr(unique_quotes["Vehicle_Value"])
(unique_quotes["Tax"]/unique_quotes["Vehicle_Value"]).mean()
unique_quotes["Tax"].corr(unique_quotes["Vehicle_Age"]) #correlation with age is very low even if age is part of the criteria for the tax
unique_quotes[["Vehicle_Age", "Vehicle_Value", "Tax", "Vehicle_Annual_Mileage"]].corr()
unique_quotes.drop(columns=["Tax"], inplace=True)
#annual mileage has a relatively big correlation with value and tax (0.48 and 0.35 resp.) Everything else is low

#%%
any(unique_quotes["Credit_Score"].isnull()), unique_quotes["Credit_Score"].dtypes #no missing values, all numbers

#%%
# I'm ignoring days to inception, I understand this is the amount of days before the insurance if effective. 
unique_quotes["Days_to_Inception"].min(),unique_quotes["Days_to_Inception"].max()
# it's always between 0 and 30 days, probably related to starting a fixed day of the month, I think it's not relevant
unique_quotes.drop(columns=["Days_to_Inception"],inplace=True)

#%%
#premium means the amount to be paid. I'm assuming monthly just means that the amount is paid montly but it corresponds to a year as well, so the payment type may only be relevant in terms of interest rate, I could leave that out, but It can also be included
unique_quotes[unique_quotes["Premium"] != unique_quotes["Capped_Premium"]]["Capped_Premium"].unique()
# what I understand by capped premium is a limit to renewal rate. It is applied only when the premium exceeds the cap, so it is not relevant for the analysis
unique_quotes["Capped_Premium"].max() #there is a maximum of 4000 #https://www.casact.org/sites/default/files/presentation/rpm_2012_handouts_session_4564_presentation_923_0.ppt#:~:text=What%20Is%20Rate%20Capping%3F,-5&text=Under%20rate%20capping%2C%20a%20customer,Premium%20at%20first%20renewal%3A%20%241200
unique_quotes.groupby("Payment_Type")["Premium"].std() #very high std, there are very large values
unique_quotes.drop(columns=["Capped_Premium"], inplace=True)
#%%
import matplotlib.pyplot as plt
#fig, ax = plt.subplots(figsize=(10,8))
#unique_quotes.boxplot(column="Premium", by="Payment_Type", ax=ax) #there are many outliers above
q3_month = np.quantile(unique_quotes[unique_quotes["Payment_Type"] == "Monthly"]["Premium"], 0.75)
unique_quotes[(unique_quotes["Payment_Type"] == "Monthly") & (unique_quotes["Premium"] > q3_month)]["Premium"].count(), unique_quotes[unique_quotes["Payment_Type"] == "Monthly"]["Premium"].count() #about 25% are considered to be outliers, that's a lot

q3_annual = np.quantile(unique_quotes[unique_quotes["Payment_Type"] == "Annual"]["Premium"], 0.75)
unique_quotes[(unique_quotes["Payment_Type"] == "Annual") & (unique_quotes["Premium"] > q3_annual)]["Premium"].count(), unique_quotes[unique_quotes["Payment_Type"] == "Annual"]["Premium"].count() #similar proportion here
#unique_quotes.boxplot(column="Premium", by="Payment_Type", ax=ax, showfliers = False)

unique_quotes.groupby("Payment_Type")["Premium"].plot(kind='kde', legend=True)
#similar distributions, monthly just a bit shifted to the right, there are many high values but thery are very spread so the distribution is only slightly skewed to the right so the tail doesn't look large
# Probably shouldn't consider the "outliers" to be outliers, this phenomena is common with prices https://stats.stackexchange.com/questions/317836/how-to-deal-when-you-have-too-many-outliers
# At most, consider using zscore to remove them https://datascience.stackexchange.com/questions/69519/how-to-tackle-too-many-outliers-in-dataset

#%%
unique_quotes
data = unique_quotes.drop(columns=["Quote_ID", "Quote_Date"])
data
#%%
#!pip install statsmodels
#https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
#X = add_constant(data.select_dtypes(include=np.number)) #there are nans
X = add_constant(data[["Driver1_Age", "Driver1_Licence_Years", "Driver1_Claims", "Vehicle_Age", "Vehicle_Value", "Vehicle_Annual_Mileage", "Credit_Score"]])
pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns) 
#All the above numercal values have VIF below 2, so they don't seem to be redundant. I can still consider replacing Age by age since licence

X["Driver1_Age"] -= X["Driver1_Licence_Years"]
pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns) #the VIF is now even lower for those two variables
# Vehicle value is highly correlated with Vehicle age, at least for single drivers, according to what I've seen on colab

# %%
data["Driver1_Age"] -= data["Driver1_Licence_Years"]
data.rename(columns={"Driver1_Age":"Driver1_Age_Start_Licence"}, inplace=True)
#%%
# finally in order to use decision tree I'm filling with -1 where there is no driver and with mean the rest
mean_years = data["Driver2_Licence_Years"].mean()
def replace_years(row):
    if (pd.isna(row['Driver2_Licence_Type']) and pd.isna(row['Driver2_Licence_Years'])):
        return -1
    elif pd.isna(row['Driver2_Licence_Years']):
        return mean_years 
    else:
        return row['Driver2_Licence_Years']

data["Driver2_Licence_Years"] = data.apply(replace_years, axis=1)
data
#%%
data["Driver1_Convictions"]=pd.get_dummies(data["Driver1_Convictions"], drop_first=True)
data["Driver1_Convictions"]
data

#%%
data["Payment_Type"]=pd.get_dummies(data["Payment_Type"], drop_first=True)
data.rename(columns={data.columns[-2]:"Monthly"}, inplace=True)
data
#%% 
#Let's do PCA
#!pip install scikit-learn
#https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
x = data[["Driver1_Age", "Driver1_Licence_Years", "Driver1_Claims", "Vehicle_Age", "Vehicle_Value", "Vehicle_Annual_Mileage", "Credit_Score"]]
x = StandardScaler().fit_transform(x)
pca = PCA(0.8) #percentage of variance I want. I can also specify n_components
principalComponents = pca.fit_transform(x)
pca.components_, pca.explained_variance_ #to get 90% of the variance I need 6 out of 7 variables and for 80% I need 5, so this is not going to be very useful
#%%
pd.get_dummies(df["Payment_Type"])
df["Payment_Type"].unique()
pd.get_dummies(df["Driver2_Licence_Type"], dummy_na=True, drop_first=True) #https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html

#%%
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing

#I need to encode labels, maybe using dummies would have been fine
# https://stackoverflow.com/questions/30384995/randomforestclassfier-fit-valueerror-could-not-convert-string-to-float
le = preprocessing.LabelEncoder()
encoded_columns = ["Driver1_Licence_Type", "Driver2_Licence_Type", "Driver1_Marital_Status"]
for column in encoded_columns:
    #i will need to keep the original values using le.classes_ to know the order
    data[column] = le.fit_transform(data[column]) 
data
#regressor.fit(X_train,y_train)

#%%

X,y = data.iloc[:,:-1], data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train,y_train)
#%%
# I tried changing max_depth but the deepter the more overfit and the shallower the more underfit, never good
test_scores = []
train_scores = []
for alpha in np.linspace(0,2,20):
    regressor = DecisionTreeRegressor(random_state=0, ccp_alpha=alpha)
    regressor.fit(X_train,y_train)
    test_scores.append(regressor.score(X_test,y_test))
    train_scores.append(regressor.score(X_train,y_train))

plt.plot(np.linspace(0,2,20), test_scores, label = "test")
plt.plot(np.linspace(0,2,20), train_scores, label = "train")
plt.legend()
#from sklearn.model_selection import cross_val_score
#cross_val_score(regressor, X_train, y_train, cv=10)
#regressor.decision_path(data.iloc[:,:-1].head(1))
#%%

#%%
regressor = DecisionTreeRegressor(random_state=0)
path = regressor.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
#%%
import matplotlib.pyplot as plt
min(ccp_alphas), max(ccp_alphas)
alphas = [alpha for alpha in ccp_alphas if 0 < alpha < 0.05][::1000]
plt.plot(np.arange(len(alphas)), alphas)

#%% 

regs = []
for ccp_alpha in ccp_alphas[::1000]:
    reg = DecisionTreeRegressor(random_state=0, ccp_alpha=ccp_alpha)
    reg.fit(X_train, y_train)
    regs.append(reg)
  
acc_scores = [reg.score(X_test,y_test) for reg in regs]

#tree_depths = [reg.tree_.max_depth for reg in regs]
#%%
plt.figure(figsize=(10,  6))
plt.grid()
plt.plot(ccp_alphas[::1000], acc_scores)
plt.xlabel("effective alpha")
plt.ylabel("Accuracy scores")
#ccp_alphas[-1]
#%%
regressor = DecisionTreeRegressor(random_state=0, ccp_alpha=4000) #ccp_alpha is not helping either overfitting or bad fitting
regressor.fit(X_train,y_train)
regressor.score(X_test,y_test), regressor.score(X_train,y_train)
#%%
duplicated_quotes
#%%
#Let us test the duplicates. I need to do the same preprocessing
# Probably I can just give the values but I deleted the quote ID so...
# It is good to review what I have done anyway
duplicated_quotes = duplicated_quotes[duplicated_quotes["Premium"] < 4000] #let's filter a bit
y_d = duplicated_quotes["Premium"]
duplicated_quotes["Driver1_DOB"] = pd.to_datetime(duplicated_quotes["Driver1_DOB"], dayfirst=True).apply(lambda x: relativedelta(datetime(2021,1,1),x).years)

duplicated_quotes["Driver1_Convictions"].replace('-9999','No', inplace=True)
duplicated_quotes["Driver1_Marital_Status"] = duplicated_quotes["Driver1_Marital_Status"].apply(lambda x: x.title())
duplicated_quotes.drop(columns=["Quote_ID", "Quote_Date","Tax","Days_to_Inception","Capped_Premium", "Premium"], inplace=True)
duplicated_quotes["Driver1_DOB"] -= duplicated_quotes["Driver1_Licence_Years"]
duplicated_quotes["Driver2_Licence_Years"] = duplicated_quotes.apply(replace_years, axis=1)
duplicated_quotes[["Driver1_Convictions","Payment_Type"]]=pd.get_dummies(duplicated_quotes[["Driver1_Convictions","Payment_Type"]], drop_first=True)
duplicated_quotes.rename(columns={"Driver1_DOB":"Driver1_Age_Start_Licence","Driver1_Age":"Driver1_Age_Start_Licence","Payment_Type":"Monthly"}, inplace=True)

encoded_columns = ["Driver1_Licence_Type", "Driver2_Licence_Type", "Driver1_Marital_Status"]
for column in encoded_columns:
    duplicated_quotes[column] = le.fit_transform(duplicated_quotes[column]) 

duplicated_quotes

#%%
import matplotlib.pyplot as plt
plt.plot(np.abs(y_d-regressor.predict(duplicated_quotes))/y_d) 
print(np.mean(np.abs(y_d-regressor.predict(duplicated_quotes))/y_d)) # The errores are too large
#%%
from scipy import stats
zscores = stats.zscore(data["Premium"])
print(zscores[zscores<3].shape,zscores[zscores>=3].shape)
data = data[np.abs(stats.zscore(data["Premium"]))<3]
data["Premium"].plot.density()
#%%
np.abs(stats.zscore(data["Premium"]))
#%%
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
data_ = data[np.abs(stats.zscore(data["Premium"]))<3]#.loc[:,(data.columns!="Driver1_Marital_Status") & (data.columns!="Driver2_Licence_Type") & (data.columns!="Driver2_Licence_Years") & (data.columns!="Driver1_Licence_Type")] #a desper
#data = data[(data["Premium"] < 1600)]
X,y = data_.iloc[:,:-1], data_.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
y_train.max(), y_test.max() # trying to make sure that the test has a wider range
#%%
regr = RandomForestRegressor(max_depth=20, n_estimators=300, max_features=4) #there's is not way this performs better on training and I don't know why
regr.fit(X_train, y_train)
regr.score(X_train, y_train), regr.score(X_test, y_test)
#%%
#Let's try to choose parameters
le = preprocessing.LabelEncoder()
encoded_columns = ["Driver1_Licence_Type", "Driver2_Licence_Type", "Driver1_Marital_Status"]
for column in encoded_columns:
    #i will need to keep the original values using le.classes_ to know the order
    data[column] = le.fit_transform(data[column]) 
data
#%%
data_rf = data[np.abs(stats.zscore(data["Premium"]))<3]
X,y = data_rf.iloc[:,:-1], data_rf.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
test_scores = []
train_scores = []
for depth in range(2,10):
    regr = RandomForestRegressor(max_depth=depth)
    regr.fit(X_train,y_train)
    test_scores.append(regr.score(X_test,y_test))
    train_scores.append(regr.score(X_train,y_train))

plt.plot(range(2,10), test_scores, label="test")
plt.plot(range(2,10), train_scores,label="train")
plt.legend() #best depth 5-6, let's set to 6
#%%
test_scores = []
train_scores = []
for min_split in np.arange(0.01,0.1, 0.01):
    regr = RandomForestRegressor(max_depth=6, min_samples_split=min_split)
    regr.fit(X_train,y_train)
    test_scores.append(regr.score(X_test,y_test))
    train_scores.append(regr.score(X_train,y_train))
#%%
plt.plot(np.arange(0.01,0.1, 0.01), test_scores, label="test")
plt.plot(np.arange(0.01,0.1, 0.01), train_scores,label="train")
plt.legend() #best split at 0.04
#%%
test_scores = []
train_scores = []
for nodes in range(2,50):
    regr = RandomForestRegressor(max_depth=6, min_samples_split=0.04, max_leaf_nodes=nodes)
    regr.fit(X_train,y_train)
    test_scores.append(regr.score(X_test,y_test))
    train_scores.append(regr.score(X_train,y_train))
plt.plot(range(2,50), test_scores, label="test")
plt.plot(range(2,50), train_scores,label="train")
plt.legend() #best about 12

#%%
test_scores = []
train_scores = []
for min_leaf in [100,150,200,250,300,350,400]:
    regr = RandomForestRegressor(max_depth=6, min_samples_split=0.04, max_leaf_nodes=12, min_samples_leaf=min_leaf)
    regr.fit(X_train,y_train)
    test_scores.append(regr.score(X_test,y_test))
    train_scores.append(regr.score(X_train,y_train))
plt.plot([100,150,200,250,300,350,400], test_scores, label="test")
plt.plot([100,150,200,250,300,350,400], train_scores,label="train")
plt.legend() #best about 250

#%%
test_scores = []
train_scores = []
for estimators in np.arange(50,150,25):
    regr = RandomForestRegressor(max_depth=6, min_samples_split=0.04, max_leaf_nodes=12, min_samples_leaf=250, n_estimators=estimators)
    regr.fit(X_train,y_train)
    test_scores.append(regr.score(X_test,y_test))
    train_scores.append(regr.score(X_train,y_train))
plt.plot(np.arange(50,150,25), test_scores, label="test")
plt.plot(np.arange(50,150,25), train_scores,label="train")
plt.legend() #best about 100

#%%
test_scores = []
train_scores = []
for samples in np.arange(0.05,0.9,0.05):
    regr = RandomForestRegressor(max_depth=6, min_samples_split=0.04, max_leaf_nodes=12, min_samples_leaf=250, n_estimators=10, max_samples=samples)
    regr.fit(X_train,y_train)
    test_scores.append(regr.score(X_test,y_test))
    train_scores.append(regr.score(X_train,y_train))
plt.plot(np.arange(0.05,0.9,0.05), test_scores, label="test")
plt.plot(np.arange(0.05,0.9,0.05), train_scores,label="train")
plt.legend() #best about 0.4

#%%
test_scores = []
train_scores = []
for features in range(1,14):
    regr = RandomForestRegressor(max_depth=6, min_samples_split=0.04, max_leaf_nodes=12, min_samples_leaf=250, n_estimators=10, max_samples=0.4, max_features=features)
    regr.fit(X_train,y_train)
    test_scores.append(regr.score(X_test,y_test))
    train_scores.append(regr.score(X_train,y_train))
plt.plot(range(1,14), test_scores, label="test")
plt.plot(range(1,14), train_scores,label="train")
plt.legend() #best about 11

#In any case the performance is very bad
#%%
from sklearn.model_selection import GridSearchCV
data_rf = data
X,y = data_rf.iloc[:,:-1], data_rf.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
regr = RandomForestRegressor()
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [2, 5, 7, 9]
}
grid_clf = GridSearchCV(regr, param_grid, cv=10)
grid_clf.fit(X_train, y_train)

#%%
regr.score(X_test,y_test), regr.score(X_train,y_train)
#%%
#plt.plot(np.abs(y_d-regr.predict(duplicated_quotes))/y_d) 
print(np.mean(np.abs(y_d-regr.predict(duplicated_quotes))/y_d))

plt.scatter(np.arange(len(y_d)),y_d)
plt.scatter(np.arange(len(y_d)),regr.predict(duplicated_quotes))

#%%
#!pip install seaborn
import seaborn as sns
#print(np.mean(np.abs(y_test-regr.predict(X_test))/y_test))

plt.scatter(np.arange(len(y_test)),y_test)
plt.scatter(np.arange(len(y_test)),regr.predict(X_test))
prediction = regr.predict(X_test)
sns.kdeplot(prediction, label = "predicted")
sns.kdeplot(y_test, label = "actual")
plt.legend()
# %%
data[(data["Premium"] < 1000) & (data["Premium"] > 400)].shape #90% of data removing higher than 1600, 80% if in addition we remove less than 400
# reducing the data doesn't help, the algorithms tends to central values
# check whether it is predicting better the quotes with central values
# try multilinear regression

#Random forest fails to predict outside the range of training data https://neptune.ai/blog/random-forest-regression-when-does-it-fail-and-why (I could try stacking or a neural network as well)
#let's go for multilinear regression. For this I should set the missing values in driver2 years to the mean

#%%

X = data #[(data["Premium"] < 1000) & (data["Premium"] > 400)]
m = X[X["Driver2_Licence_Years"]>-1]["Driver2_Licence_Years"].mean()
X["Driver2_Licence_Years"] = X["Driver2_Licence_Years"].apply(lambda x: m if x < 0 else x)
#%%
def standarize(df):
    return (df-df.mean())/df.std()
# standarizing is not essential but helps interpretability for numerical variables. https://stats.stackexchange.com/questions/29781/when-conducting-multiple-regression-when-should-you-center-your-predictor-varia
#  For categorical variables does not make much sense https://stats.stackexchange.com/questions/116560/scaling-categorical-data-in-regression
X[["Driver1_Age_Start_Licence", "Driver1_Licence_Years", "Driver2_Licence_Years", "Driver1_Claims", "Vehicle_Age", "Vehicle_Value", "Vehicle_Annual_Mileage", "Credit_Score"]]=standarize(X[["Driver1_Age_Start_Licence", "Driver1_Licence_Years", "Driver2_Licence_Years", "Driver1_Claims", "Vehicle_Age", "Vehicle_Value", "Vehicle_Annual_Mileage", "Credit_Score"]])
#%%
from sklearn.linear_model import LinearRegression
X,y = X.iloc[:,:-1], X.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg = LinearRegression().fit(X, y)
reg.score(X_test,y_test), reg.score(X_train,y_train) #terrible performance but equally bad on both cases
#consider polynomial model
#%%
from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.1)
clf.fit(X_train,y_train)
clf.score(X_test,y_test),clf.score(X_train,y_train) #similar, alpha is not doing much when small and making it worse when big
#%%
#plt.scatter(X["Driver1_Age_Start_Licence"], X["Premium"])
#plt.bar(X["Driver1_Licence_Type"], X.groupby("Driver1_Licence_Type").mean()["Premium"])
#plt.scatter(X["Driver1_Licence_Type"], X["Premium"])
fig, axs = plt.subplots(len(X.columns)-1, figsize = (6,25))
for i in range(len(X.columns[:-1])):
    axs[i].scatter(X[X.columns[i]], X["Premium"])
    axs[i].set_title(X.columns[i])


#%%
from scipy import stats
data_ = data[np.abs(stats.zscore(data["Premium"]))<3]
for column in data_.columns[:-1]:
    print(f"{column} correlation:{data_[column].corr(data_['Premium'])}")

#correlations are generally very low individually, only driver1years is above 0.1 with -0.34

data_
#%%
data_["Vehicle_Value"].min(),data_["Vehicle_Annual_Mileage"].min(), data_["Vehicle_Age"].min() #I missed some negative values here fuck!
data_[data_["Vehicle_Value"]<0].count(),data_[data_["Vehicle_Annual_Mileage"]<0].count(), data_[data_["Vehicle_Age"]<0].count() #A few hundreads, can be droped
data_ = data_[(data_["Vehicle_Value"]>=0) & (data_["Vehicle_Annual_Mileage"]>=0) & (data_["Vehicle_Age"]>=0)]
data_ = data_[(data_["Vehicle_Age"]<75) & (data_["Vehicle_Annual_Mileage"]<200000)]
#mean_value= data_[data_["Vehicle_Value"] >=0]["Vehicle_Value"].mean()
#mean_age= data_[data_["Vehicle_Age"] >=0]["Vehicle_Age"].mean()
#mean_mileage= data_[data_["Vehicle_Annual_Mileage"] >=0]["Vehicle_Annual_Mileage"].mean()
#data_["Vehicle_Value"] = data_["Vehicle_Value"].apply(lambda x: x if x>=0 else mean_value)
#data_["Vehicle_Annual_Mileage"] = data_["Vehicle_Annual_Mileage"].apply(lambda x: x if x>=0 else mean_mileage)
#data_["Vehicle_Age"] = data_["Vehicle_Age"].apply(lambda x: x if x>=0 else mean_age)
#%%

#%%
import matplotlib.pyplot as plt
fig, axs = plt.subplots(len(data_.columns)-1, figsize = (6,25))
for i in range(len(data_.columns[:-1])):
    axs[i].scatter(data_ [data_.columns[i]], data_["Premium"])
    axs[i].set_title(data_.columns[i])
#%%
# I'll see if this works at least for algebraic non-outliers
q3 = np.quantile(data_["Premium"], 0.75)
data_q3 = data_[data_["Premium"] < q3]
X,y = data_q3.iloc[:,:-1], data_q3.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
y_train.max(), y_test.max() # trying to make sure that the test has a wider range
#%%
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=6, min_samples_split=0.04, max_leaf_nodes=12, min_samples_leaf=250, n_estimators=200) # terrible wtf so outliers is not really the problem
regr.fit(X_train, y_train)
regr.score(X_train, y_train), regr.score(X_test, y_test)
#%%
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X, y)
reg.score(X_test,y_test), reg.score(X_train,y_train) #very bad
#%%
fig, axs = plt.subplots(len(data_q3.columns)-1, figsize = (6,25))
for i in range(len(data_q3.columns[:-1])):
    axs[i].scatter(data_q3 [data_q3.columns[i]], data_q3["Premium"])
    axs[i].set_title(data_q3.columns[i])
# the data is all over the place
#%%
sorted(data_["Vehicle_Age"].unique()) #there is a jump from 36 to 75 
data_[data_["Vehicle_Age"]==75].count() #there are only 168, so I may drop them
sorted(data_["Vehicle_Annual_Mileage"].unique())[-2:] #jums from 50000 to 200000
data_[data_["Vehicle_Annual_Mileage"]==200000].count() #only 67, will drop them
#%%

data_ = data_[(data_["Vehicle_Age"]<75) & (data_["Vehicle_Annual_Mileage"]<200000)]
q3 = np.quantile(data_["Premium"], 0.75)
data_q3 = data_[data_["Premium"] < q3]
fig, axs = plt.subplots(len(data_q3.columns)-1, figsize = (6,25))
for i in range(len(data_q3.columns[:-1])):
    axs[i].scatter(data_q3[data_q3.columns[i]], data_q3["Premium"])
    axs[i].set_title(data_q3.columns[i])

#%%

X,y = data_q3.iloc[:,:-1], data_q3.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
y_train.max(), y_test.max()
#%% 
regr = RandomForestRegressor(max_depth=20, n_estimators=300, max_features=5) # terrible wtf so outliers is not really the problem
regr.fit(X_train, y_train)
regr.score(X_train, y_train), regr.score(X_test, y_test)
# %%
# I give up with random forest. I don't think it really needs to extrapolate if I make sure the the test has a wider range. Maybe it's because of the missing data. Last attempt will be a neural network, and I'll present whatever fits best
