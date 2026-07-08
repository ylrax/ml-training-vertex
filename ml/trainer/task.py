#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import RandomizedSearchCV


# In[2]:


import bigframes.pandas as bpd


# In[3]:


bpd.options.bigquery.project = "vertexai-ml-pipelines"
bpd.options.bigquery.location = "europe-west1"
bpd.options.display.progress_bar = None


# In[4]:


# This is how you read a BigQuery table
train = bpd.read_gbq("vertexai-ml-pipelines.ml_training_data.house-prices-train").to_pandas()
test = bpd.read_gbq("vertexai-ml-pipelines.ml_training_data.house-prices-test").to_pandas()


# In[5]:


train.set_index('Id', inplace=True)
test.set_index('Id', inplace=True)
#train.drop(columns='Id', inplace=True)
#test.drop(columns='Id', inplace=True)


# In[6]:


train.head()


# In[7]:


train.iloc[0]


# In[8]:


print("train: ", train.shape)
print("test: ", test.shape)


# In[9]:


X = pd.concat([train.drop("SalePrice", axis=1),test], axis=0)
y = train[['SalePrice']]


# In[10]:


numeric_ = X.select_dtypes(exclude=['object']).drop(['MSSubClass'], axis=1).copy()
numeric_.columns


# In[11]:


disc_num_var = ['OverallQual','OverallCond','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath',
                'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'MoSold', 'YrSold']

cont_num_var = []
for i in numeric_.columns:
    if i not in disc_num_var:
        cont_num_var.append(i)


# In[12]:


cat_train = X.select_dtypes(include=['object']).copy()
cat_train['MSSubClass'] = X['MSSubClass']   #MSSubClass is nominal
cat_train.columns


# In[13]:

print("numeric extraction")
try:
    numeric_train = train.select_dtypes(exclude=['object', 'str'])
except:
    numeric_train = train.select_dtypes(exclude=['object', 'string'])
correlation = numeric_train.corr()
correlation[['SalePrice']].sort_values(['SalePrice'], ascending=False)


# In[14]:


X.drop(['GarageYrBlt','TotRmsAbvGrd','1stFlrSF','GarageCars'], axis=1, inplace=True)


# In[15]:


X.drop(['PoolQC','MiscFeature','Alley'], axis=1, inplace=True)


# In[16]:


correlation[['SalePrice']].sort_values(['SalePrice'], ascending=False).tail(10)

X.drop(['MoSold','YrSold'], axis=1, inplace=True)


# In[17]:


cat_col = X.select_dtypes(include=['object']).columns
overfit_cat = []
for i in cat_col:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 96:
        overfit_cat.append(i)

overfit_cat = list(overfit_cat)
X = X.drop(overfit_cat, axis=1)


# In[18]:


num_col = X.select_dtypes(exclude=['object']).drop(['MSSubClass'], axis=1).columns
overfit_num = []
for i in num_col:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 96:
        overfit_num.append(i)

overfit_num = list(overfit_num)
X = X.drop(overfit_num, axis=1)


# In[19]:


print("Categorical Features with >96% of the same value: ",overfit_cat)
print("Numerical Features with >96% of the same value: ",overfit_num)


# In[20]:


train['LotFrontage'] = pd.to_numeric(train['LotFrontage'], errors='coerce')


# In[21]:


train = train.drop(train[train['LotFrontage'] > 200].index)
train = train.drop(train[train['LotArea'] > 100000].index)
train = train.drop(train[train['BsmtFinSF1'] > 4000].index)
train = train.drop(train[train['TotalBsmtSF'] > 5000].index)
train = train.drop(train[train['GrLivArea'] > 4000].index)


# In[22]:


print(X.shape)


# In[23]:


cat = ['GarageType','GarageFinish','BsmtFinType2','BsmtExposure','BsmtFinType1', 
       'GarageCond','GarageQual','BsmtCond','BsmtQual','FireplaceQu','Fence',"KitchenQual",
       "HeatingQC",'ExterQual','ExterCond']

X[cat] = X[cat].fillna("NA")


# In[24]:


#categorical
cols = ["MasVnrType", "MSZoning", "Exterior1st", "Exterior2nd", "SaleType", "Electrical", "Functional"]
#X[cols] = X.groupby("Neighborhood")[cols].transform(lambda x: x.fillna(x.mode()[0]))
X[cols] = X.groupby("Neighborhood")[cols].transform(lambda s: s.fillna(s.mode().iloc[0]) if not s.mode().empty else s)


# In[25]:


for c in ["BsmtHalfBath", "BsmtFullBath", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "MasVnrArea", 'LotFrontage', 'GarageArea']:
    X[c] = pd.to_numeric(X[c], errors='coerce').astype('Float64')
#X['GarageArea'] = pd.to_numeric(X['GarageArea'], errors='coerce').astype('Float64')


# In[26]:


print("Mean of LotFrontage: ", X['LotFrontage'].mean())
print("Mean of GarageArea: ", X['GarageArea'].mean())


# In[27]:


#for correlated relationship
X['LotFrontage'] = X.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))
X['GarageArea'] = X.groupby('Neighborhood')['GarageArea'].transform(lambda x: x.fillna(x.mean()))
X['MSZoning'] = X.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

#numerical
cont = ["BsmtHalfBath", "BsmtFullBath", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "MasVnrArea"]
X[cont] = X[cont].fillna(X[cont].mean())


# In[28]:


X['MSSubClass'] = X['MSSubClass'].apply(str)


# In[29]:


ordinal_map = {'Ex': 5,'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA':0}
fintype_map = {'GLQ': 6,'ALQ': 5,'BLQ': 4,'Rec': 3,'LwQ': 2,'Unf': 1, 'NA': 0}
expose_map = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}
fence_map = {'GdPrv': 4,'MnPrv': 3,'GdWo': 2, 'MnWw': 1,'NA': 0}


# In[30]:


ord_col = ['ExterQual','ExterCond','BsmtQual', 'BsmtCond','HeatingQC','KitchenQual','GarageQual','GarageCond', 'FireplaceQu']
for col in ord_col:
    X[col] = X[col].map(ordinal_map)

fin_col = ['BsmtFinType1','BsmtFinType2']
for col in fin_col:
    X[col] = X[col].map(fintype_map)

X['BsmtExposure'] = X['BsmtExposure'].map(expose_map)
X['Fence'] = X['Fence'].map(fence_map)


# In[31]:


X['TotalLot'] = X['LotFrontage'] + X['LotArea']
X['TotalBsmtFin'] = X['BsmtFinSF1'] + X['BsmtFinSF2']
X['TotalSF'] = X['TotalBsmtSF'] + X['2ndFlrSF']
X['TotalBath'] = X['FullBath'] + X['HalfBath']
X['TotalPorch'] = X['OpenPorchSF'] + X['EnclosedPorch'] + X['ScreenPorch']


# In[32]:


colum = ['MasVnrArea','TotalBsmtFin','TotalBsmtSF','2ndFlrSF','WoodDeckSF','TotalPorch']

for col in colum:
    col_name = col+'_bin'
    X[col_name] = X[col].apply(lambda x: 1 if x > 0 else 0)


# In[33]:


X = pd.get_dummies(X)


# In[34]:


y["SalePrice"] = np.log(y['SalePrice'])


# In[35]:


x = X.loc[train.index]
y = y.loc[train.index]
test = X.loc[test.index]


# In[36]:


x.loc[609]


# In[37]:


x


# In[38]:


cols = x.select_dtypes(np.number).columns
transformer = RobustScaler().fit(x[cols])
x[cols] = transformer.transform(x[cols])
test[cols] = transformer.transform(test[cols])


# In[39]:


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=2020)


# In[42]:



from sklearn.metrics import mean_squared_error, mean_absolute_error
#from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
#from catboost import CatBoostRegressor


# In[44]:


lgbm = LGBMRegressor(boosting_type='gbdt',objective='regression', max_depth=-1,
                    lambda_l1=0.0001, lambda_l2=0, learning_rate=0.1,
                    n_estimators=100, max_bin=200, min_child_samples=20, 
                    bagging_fraction=0.75, bagging_freq=5,
                    bagging_seed=7, feature_fraction=0.8,
                    feature_fraction_seed=7, verbose=-1)


# In[45]:


param_lst = {
    'max_depth' : [2, 5, 8, 10],
    'learning_rate' : [0.001, 0.01, 0.1, 0.2],
    'n_estimators' : [100, 300, 500, 1000, 1500],
    'lambda_l1' : [0.0001, 0.001, 0.01],
    'lambda_l2' : [0, 0.0001, 0.001, 0.01],
    'feature_fraction' : [0.4, 0.6, 0.8],
    'min_child_samples' : [5, 10, 20, 25]
}

lightgbm = RandomizedSearchCV(estimator = lgbm, param_distributions = param_lst,
                              n_iter = 100, scoring = 'neg_root_mean_squared_error',
                              cv = 5)

lightgbm_search = lightgbm.fit(X_train, y_train)

# LightBGM with tuned hyperparameters
best_param = lightgbm_search.best_params_
lgbm = LGBMRegressor(**best_param)


# #### Training and Evaluation

# In[46]:


def mean_cross_val(model, X, y):
    score = cross_val_score(model, X, y, cv=5)
    mean = score.mean()
    return mean

lgbm.fit(X_train, y_train)   
preds = lgbm.predict(X_val) 
preds_test_lgbm = lgbm.predict(test)
mae_lgbm = mean_absolute_error(y_val, preds)
rmse_lgbm = np.sqrt(mean_squared_error(y_val, preds))
score_lgbm = lgbm.score(X_val, y_val)
cv_lgbm = mean_cross_val(lgbm, x, y)


# In[47]:


model_performances = pd.DataFrame({
    "Model" : ["LGBM"],
    "CV(5)" : [str(cv_lgbm)[0:5], ],
    "MAE" :   [str(mae_lgbm)[0:5],],
    "RMSE" :  [str(rmse_lgbm)[0:5]],
    "Score" : [str(score_lgbm)[0:5]]
})

print("Sorted by Score:")
print(model_performances.sort_values(by="Score", ascending=False))


# In[49]:


#lgbm.save("/gcs/models-ml-testing-files/model-output")


# In[56]:


# Save the native booster to a text file
#lgbm.booster_.save_model('/gcs/models-ml-testing-files/model-output/lgbm_model.txt')


# In[57]:


from google.cloud import storage
import joblib

# 1. Save locally
local_temp_path = '/tmp/lgbm_model.pkl'
joblib.dump(lgbm, local_temp_path)

# 2. Upload to GCS
client = storage.Client()
bucket = client.bucket('models-ml-testing-files')
blob = bucket.blob('model-output/lgbm_model.pkl')

blob.upload_from_filename(local_temp_path)
