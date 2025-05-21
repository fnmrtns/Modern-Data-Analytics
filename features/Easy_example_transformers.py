
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


### Load the data

zip_path = "./Data/cordis-HORIZONprojects-xlsx.zip"
csv_filename = "project.xlsx"

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
legal_basis_dep.groupby(['legalBasis','title']).size()

df.shape
df1 = df.merge(legal_basis_dep,left_on = 'id',right_on = 'projectID',how = 'left')
df1.isna().sum()
df1.shape

df1[['title_x','title_y']]

## Get top topics
aux = df1.groupby('title_y').size().reset_index(name = 'n').sort_values(by = 'n',ascending = False)
aux.shape
aux['cumsum'] = aux['n'].cumsum()
aux['cumsum_per'] = aux['cumsum']/aux['n'].sum()
aux1 = aux.query(f'cumsum_per < 0.9')
aux1.shape

df1['topic_def'] = np.where(df1['title_y'].isin(aux1['title_y']),df1['title_y'],'Other')
df1.groupby('topic_def').size()
df1.shape

##################
### merge2 - organizations
##################

len(organization['projectID'].unique())
organization.query(f'role == "coordinator"').shape

organization_num = organization.groupby('projectID').size().reset_index(name = 'num_org')
organization_num.shape

organization_country = organization.query(f'role == "coordinator"')[['projectID','country']]
organization_country.shape

aux = organization_country.groupby(['country']).size().reset_index(name = 'n').sort_values(by = 'n',ascending = False)
aux['cumsum'] = aux['n'].cumsum()
aux['cumsum_per'] = aux['cumsum']/aux['n'].sum()
aux1 = aux.query(f'cumsum_per < 0.9')

df1.shape
df1 = df1.merge(organization_num,left_on = 'id',right_on = 'projectID',how = 'left')
df1.shape
df1 = df1.merge(organization_country,left_on = 'id',right_on = 'projectID',how = 'left')
df1.shape

df1['country1'] = np.where(df1['country'].isin(aux1['country']),df1['country'],'Other')
df1.columns
#df['title_y']

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


### Transformations

columns_to_scale = ['totalCost','ecMaxContribution','ratio','pry_duration_m','pry_duration_d1','delay_d1','num_org']
columns_to_encode = ['title_y','country1','cost0']

transformer = ColumnTransformer(
    transformers=[
        ('scale',StandardScaler(),columns_to_scale),
        ('dummies',OneHotEncoder(handle_unknown='ignore'),columns_to_encode)
        ],
        remainder='drop'    
)

df = df.query(f'~delay_m.isna()')
y= df[['delay_m']]
X = df.drop(['delay_m'],axis=1,inplace=False)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)

X_train1 = transformer.fit_transform(X_train)
X_test1 = transformer.fit_transform(X_test)

##################
### Model
##################

# Example
model = LinearRegression()
model.fit(X_train1,y_train)
# Print the coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

# Make predictions
y_pred = model.predict(X_test1)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")


