

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
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import os



zip_path = "./Data/cordis-HORIZONprojects-xlsx.zip"
csv_filename = "project.xlsx"

# Open the ZIP file and read the CSV directly
# with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#     # List all files in the archive (optional, for verification)
#     print("Files in the archive:", zip_ref.namelist())
    
#     # Read the CSV
#     with zip_ref.open(csv_filename) as file:
#         df = pd.read_excel(file)

# df.head()
# len(df['id'].unique())

def load_excel_from_zip(zip_file_path):
    # Verifica que el archivo exista
    if not os.path.exists(zip_file_path):
        print(f"El archivo {zip_file_path} no existe.")
        return
    
    # Crear un diccionario para almacenar los DataFrames
    excel_files = {}
    
    # Leer archivos del ZIP
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            # Solo archivos Excel
            if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
                # Leer el archivo Excel como DataFrame
                with zip_ref.open(file_name) as excel_file:
                    df = pd.read_excel(excel_file)
                    excel_files[file_name] = df
                    print(f"Cargado: {file_name}")

    return excel_files

excel_files = load_excel_from_zip(zip_path)


# Mostrar los DataFrames cargados
for file_name, df in excel_files.items():
    print(f"\n📄 Archivo: {file_name}")
    #print(df.head())

#Check files
df = excel_files['project.xlsx']

excel_files['webLink.xlsx'] ## Nothing
legal_basis = excel_files['legalBasis.xlsx'] ## title
excel_files['topics.xlsx'] ## Nothing
organization = excel_files['organization.xlsx'] ## Here there are interesting things to work on
excel_files['webItem.xlsx'] ## Nothing
excel_files['euroSciVoc.xlsx'] ## euroSciVocTitle

##################
### merge1 -legal basis
##################

legal_basis.shape
legal_basis.query(f'uniqueProgrammePart.notna()').shape
legal_basis.columns
legal_basis.head()
len(legal_basis['projectID'].unique())
legal_basis_dep = legal_basis.query(f'uniqueProgrammePart.notna()')
legal_basis_dep.shape

df.shape
df1 = df.merge(legal_basis_dep,left_on = 'id',right_on = 'projectID',how = 'left')
df1.isna().sum()

df1[['title_x','title_y']]
aux = df1.groupby('title_y').size().reset_index(name = 'n').sort_values(by = 'n',ascending = False)
aux.shape
aux['cumsum'] = aux['n'].cumsum()
aux['cumsum_per'] = aux['cumsum']/aux['n'].sum()
aux1 = aux.query(f'cumsum_per < 0.9')
aux1.shape

df1['topic_def'] = np.where(df1['title_y'].isin(aux1['title_y']),df1['title_y'],'Other')
df1.groupby('topic_def').size()

##################
### merge2 - organizations
##################

len(organization['projectID'].unique())
organization.query(f'role == "coordinator"').shape

organization_num = organization.groupby('projectID').size().reset_index(name = 'num_org')
organization_num.shape

organization_country = organization.query(f'role == "coordinator"')[['projectID','country']]
organization_country.shape

aux = organization_country.groupby('country').size().reset_index(name = 'n').sort_values(by = 'n',ascending = False)
aux['cumsum'] = aux['n'].cumsum()
aux['cumsum_per'] = aux['cumsum']/aux['n'].sum()
aux1 = aux.query(f'cumsum_per < 0.9')

df1.shape
df1 = df1.merge(organization_num,left_on = 'id',right_on = 'projectID',how = 'left')
df1.shape
df1 = df1.merge(organization_country,left_on = 'id',right_on = 'projectID',how = 'left')

df1['country1'] = np.where(df1['country'].isin(aux1['country']),df1['country'],'Other')
df1.columns


##################
### Create important columns
##################
df = df1
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

df['ratio'] = np.where(df['totalCost'] == 0, 1 ,df['ecMaxContribution']/df['totalCost']) 
df['ratio'].describe()
df.query(f'totalCost == 0')
df.query(f'ratio > 1')

df['cost0'] = np.where(df['totalCost'] == 0, 1 ,0) 

##################
### Preprocessing 1
##################


# df.groupby('legalBasis').size() 
# df.groupby('legalBasis')['delay_m'].mean() ## Interesting
# df.columns

# df_dummies = pd.get_dummies(df[['legalBasis']]).astype(int)
# df = pd.concat([df,df_dummies],axis = 1)
# df.columns
# df.head()

# df1 = pd.concat([df[['id','totalCost','ecMaxContribution','delay_m','pry_duration_m']],df_dummies],axis=1)

# Missings? 

# df1.isna().sum()

##############
## SPLIT DF
##############
#df.groupby('topics').size().reset_index(name='n').sort_values(by='n',ascending = False)
# train_df, test_df = train_test_split(df1, test_size=0.2, random_state=42)

##############
## SCALE
##############
# train_df.columns
# train_df.shape

# train_df_scale = train_df[['totalCost','ecMaxContribution','pry_duration_m']]
# StS = StandardScaler()
# train_df_scale1 = pd.DataFrame(StS.fit_transform(train_df_scale),columns=['totalCost','ecMaxContribution','pry_duration_m'])
# train_df_scale1.shape
# train_df_scale1.isna().sum()
# train_df_scale1.reset_index(drop=True, inplace=True)

# exclude_cols = ['totalCost','ecMaxContribution','pry_duration_m']
# aux = train_df.loc[:, train_df.columns.difference(exclude_cols)]
# aux.reset_index(drop=True, inplace=True)

# train_df2 = pd.concat([aux,train_df_scale1],axis=1)
# train_df2.columns
# train_df2.shape
# train_df2 = train_df2.drop(columns=['delay_m','id'])

# train_df2.isna().sum()

##################
### Preprocessing 2
##################
df1.columns
df1['totalCost'].corr(df1['ecMaxContribution'])

columns_to_scale = ['totalCost','ratio','pry_duration_m','num_org']
columns_to_encode = ['legalBasis','country1','topic_def','cost0']

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

##################
### Model
##################

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
## Pipeline
##############

pipe = Pipeline(
    [
        ('preprocessiong',transformer),
        ('model',LinearRegression())
    ]
)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)
pipe = pipe.fit(X_train,y_train)
preds = pipe.predict(X_test)
pipe.score(X_test,y_test)

pipe1 = Pipeline(
    [
        ('preprocessiong',transformer),
        ('model',XGBRegressor(                  # Step 2: Train an XGBoost regressor
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ))
    ]
)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)
pipe1 = pipe1.fit(X_train,y_train)
preds = pipe1.predict(X_test)
pipe1.score(X_test,y_test)

##############
## Pipeline 2
##############

param_grid = {
    'regressor__n_estimators': [50, 100],
    'regressor__learning_rate': [0.01, 0.1],
    'regressor__max_depth': [3, 5],
    'regressor__subsample': [0.8, 1.0]
}

# Wrap the pipeline in GridSearchCV
grid_search = GridSearchCV(
    estimator=pipe1,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1
)

grid_search.fit(X_train, y_train)





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



