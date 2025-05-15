

import zipfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from  sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


zip_path = "./Data/cordis-HORIZONprojects-xlsx.zip"
csv_filename = "project.xlsx"

# Open the ZIP file and read the CSV directly
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # List all files in the archive (optional, for verification)
    print("Files in the archive:", zip_ref.namelist())
    
    # Read the CSV
    with zip_ref.open(csv_filename) as file:
        df = pd.read_excel(file)

df.head()

##################
### Create important columns
##################

df['startDate1'] = pd.to_datetime(df['startDate'],format="%Y-%m-%d")
df['ecSignatureDate1'] = pd.to_datetime(df['ecSignatureDate'],format="%Y-%m-%d")
df['endDate1'] = pd.to_datetime(df['endDate'],format="%Y-%m-%d")

df['delay_d'] = np.where(df['startDate1'] < df['ecSignatureDate1'],0,df['startDate1'] - df['ecSignatureDate1'])
df['delay_d1'] = pd.to_timedelta(df['delay_d']).dt.days
df['delay_m'] = (df['delay_d1']/30.44).round(2)

df['pry_duration_d'] = np.where(df['endDate1'] < df['startDate1'],0,df['endDate1'] - df['startDate1'])
df['pry_duration_d1'] = pd.to_timedelta(df['pry_duration_d']).dt.days
df['pry_duration_m'] = (df['pry_duration_d1']/30.44).round(2)

def coma_to_point (df,column):
    df[column] = pd.to_numeric( df[column].str.replace(',','.'))
    return df

df = coma_to_point(df,'totalCost')
df = coma_to_point(df,'ecMaxContribution')

##################
### Preprocessing 1
##################


df.groupby('legalBasis').size() 
df.groupby('legalBasis')['delay_m'].mean() ## Interesting
df.columns

df_dummies = pd.get_dummies(df[['legalBasis']]).astype(int)
df = pd.concat([df,df_dummies],axis = 1)
df.columns
df.head()

df1 = pd.concat([df[['id','totalCost','ecMaxContribution','delay_m','pry_duration_m']],df_dummies],axis=1)

# Missings? 

df1.isna().sum()

##############
## SPLIT DF
##############
#df.groupby('topics').size().reset_index(name='n').sort_values(by='n',ascending = False)
train_df, test_df = train_test_split(df1, test_size=0.2, random_state=42)

##############
## SCALE
##############
train_df.columns
train_df.shape

train_df_scale = train_df[['totalCost','ecMaxContribution','pry_duration_m']]
StS = StandardScaler()
train_df_scale1 = pd.DataFrame(StS.fit_transform(train_df_scale),columns=['totalCost','ecMaxContribution','pry_duration_m'])
train_df_scale1.shape
train_df_scale1.isna().sum()
train_df_scale1.reset_index(drop=True, inplace=True)

exclude_cols = ['totalCost','ecMaxContribution','pry_duration_m']
aux = train_df.loc[:, train_df.columns.difference(exclude_cols)]
aux.reset_index(drop=True, inplace=True)

train_df2 = pd.concat([aux,train_df_scale1],axis=1)
train_df2.columns
train_df2.shape
train_df2 = train_df2.drop(columns=['delay_m','id'])

train_df2.isna().sum()

##################
### Preprocessing 2
##################

columns_to_scale = ['totalCost','ecMaxContribution','pry_duration_m']
columns_to_encode = ['legalBasis']

transformer = ColumnTransformer(
    transformers=[
        ('scale',StandardScaler(),columns_to_scale),
        ('dummies',OneHotEncoder(handle_unknown='ignore'),columns_to_encode)
        ],
        remainder='drop'    
)

df.columns
#df.query(f'delay212_m.isna()')
df = df.query(f'~delay_m.isna()')
y= df['delay_m']
X = df.drop(['delay_m'],axis=1,inplace=False)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)
X_train = transformer.fit_transform(X_train)

# Example
model = LinearRegression()
model.fit(X_train,y_train)

# Print the coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

# Make predictions
X_test = transformer.fit_transform(X_test)
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

##############
## PCA
##############
pca = PCA(n_components=len(train_df2.columns))
pca.fit(train_df2)
explained_variance = pca.explained_variance_ratio_
explained_variance

plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(explained_variance), marker='o')
plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Variance Explained")
plt.show()

loadings = pd.DataFrame(
    pca.components_.T, 
    columns=[f'PC{i+1}' for i in range(len(train_df2.columns))], 
    index=train_df2.columns
)

print(loadings['PC1'].sort_values(ascending=False).head(5))

plt.figure(figsize=(12, 8))
sns.heatmap(loadings, cmap='RdBu_r', annot=True, fmt=".2f")
plt.title("Component Loadings")
plt.show()

##############
## Pipeline
##############

pipe = Pipeline(
    [
        ()
    ]
)


