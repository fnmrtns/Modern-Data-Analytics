# 📘 Kedro Project Report

## 📊 Pipelines

### `train_xgboost`

- **Node:** `xgb_preprocessing_node`
  - 📥 Inputs: model_input_table
  - 📤 Outputs: xgb_input_table
  - 🧠 Function: `preprocess_for_xgboost`

- **Node:** `xgb_split_data_node`
  - 📥 Inputs: xgb_input_table
  - 📤 Outputs: xgb_X_train, xgb_X_test, xgb_y_train, xgb_y_test
  - 🧠 Function: `split_xgboost_data`

- **Node:** `xgb_training_node`
  - 📥 Inputs: xgb_X_train, xgb_y_train, params:xgb_params
  - 📤 Outputs: xgb_model
  - 🧠 Function: `train_xgboost_model`

### `train_catboost`

- **Node:** `cb_preprocessing_node`
  - 📥 Inputs: model_input_table
  - 📤 Outputs: cb_input_table
  - 🧠 Function: `preprocess_for_catboost`

- **Node:** `cb_split_data_node`
  - 📥 Inputs: cb_input_table
  - 📤 Outputs: cb_X_train, cb_X_test, cb_y_train, cb_y_test, cb_cat_features
  - 🧠 Function: `split_catboost_data`

- **Node:** `cb_training_node`
  - 📥 Inputs: cb_X_train, cb_y_train, cb_X_test, cb_y_test, params:cb_params, cb_cat_features
  - 📤 Outputs: cb_model
  - 🧠 Function: `train_catboost_model`

### `evaluate_model`

- **Node:** `evaluate_cb_point_node`
  - 📥 Inputs: cb_model, cb_X_test, cb_y_test
  - 📤 Outputs: metrics_cb
  - 🧠 Function: `evaluate_cb_point_model`

- **Node:** `evaluate_xgb_node`
  - 📥 Inputs: xgb_model, xgb_X_test, xgb_y_test
  - 📤 Outputs: metrics_xgb
  - 🧠 Function: `evaluate_xgb_model`

### `compare_models`

- **Node:** `compare_model_metrics_node`
  - 📥 Inputs: metrics_xgb, metrics_cb
  - 📤 Outputs: model_comparison_table
  - 🧠 Function: `compare_model_metrics`

### `general_preprocessing`

- **Node:** `preprocess_node`
  - 📥 Inputs: project_raw, programme_raw, organization_raw
  - 📤 Outputs: model_input_table
  - 🧠 Function: `preprocess`

### `__default__`

- **Node:** `preprocess_node`
  - 📥 Inputs: project_raw, programme_raw, organization_raw
  - 📤 Outputs: model_input_table
  - 🧠 Function: `preprocess`

- **Node:** `cb_preprocessing_node`
  - 📥 Inputs: model_input_table
  - 📤 Outputs: cb_input_table
  - 🧠 Function: `preprocess_for_catboost`

- **Node:** `xgb_preprocessing_node`
  - 📥 Inputs: model_input_table
  - 📤 Outputs: xgb_input_table
  - 🧠 Function: `preprocess_for_xgboost`

- **Node:** `cb_split_data_node`
  - 📥 Inputs: cb_input_table
  - 📤 Outputs: cb_X_train, cb_X_test, cb_y_train, cb_y_test, cb_cat_features
  - 🧠 Function: `split_catboost_data`

- **Node:** `xgb_split_data_node`
  - 📥 Inputs: xgb_input_table
  - 📤 Outputs: xgb_X_train, xgb_X_test, xgb_y_train, xgb_y_test
  - 🧠 Function: `split_xgboost_data`

- **Node:** `cb_training_node`
  - 📥 Inputs: cb_X_train, cb_y_train, cb_X_test, cb_y_test, params:cb_params, cb_cat_features
  - 📤 Outputs: cb_model
  - 🧠 Function: `train_catboost_model`

- **Node:** `xgb_training_node`
  - 📥 Inputs: xgb_X_train, xgb_y_train, params:xgb_params
  - 📤 Outputs: xgb_model
  - 🧠 Function: `train_xgboost_model`

- **Node:** `evaluate_cb_point_node`
  - 📥 Inputs: cb_model, cb_X_test, cb_y_test
  - 📤 Outputs: metrics_cb
  - 🧠 Function: `evaluate_cb_point_model`

- **Node:** `evaluate_xgb_node`
  - 📥 Inputs: xgb_model, xgb_X_test, xgb_y_test
  - 📤 Outputs: metrics_xgb
  - 🧠 Function: `evaluate_xgb_model`

- **Node:** `compare_model_metrics_node`
  - 📥 Inputs: metrics_xgb, metrics_cb
  - 📤 Outputs: model_comparison_table
  - 🧠 Function: `compare_model_metrics`


## 📁 Data Catalog

- `project_raw`: **kedro_datasets.json.JSONDataset** → `data/01_raw/project.json`
- `organization_raw`: **kedro_datasets.json.JSONDataset** → `data/01_raw/organization.json`
- `programme_raw`: **kedro_datasets.json.JSONDataset** → `data/01_raw/programme.json`
- `model_input_table`: **kedro_datasets.pandas.ParquetDataset** → `data/03_primary/selected_features.parquet`
- `xgb_input_table`: **kedro_datasets.pandas.ParquetDataset** → `data/05_model_input/xgb_input_table.parquet`
- `xgb_X_train`: **kedro_datasets.pandas.ParquetDataset** → `data/05_model_input/xgb_X_train.parquet`
- `xgb_X_test`: **kedro_datasets.pandas.ParquetDataset** → `data/05_model_input/xgb_X_test.parquet`
- `xgb_y_train`: **kedro_datasets.pandas.ParquetDataset** → `data/05_model_input/xgb_y_train.parquet`
- `xgb_y_test`: **kedro_datasets.pandas.ParquetDataset** → `data/05_model_input/xgb_y_test.parquet`
- `xgb_model`: **pickle.PickleDataset** → `data/06_models/xgb_model.pkl`
- `cb_input_table`: **kedro_datasets.pandas.ParquetDataset** → `data/05_model_input/cb_input_table.parquet`
- `cb_X_train`: **kedro_datasets.pandas.ParquetDataset** → `data/05_model_input/cb_X_train.parquet`
- `cb_X_test`: **kedro_datasets.pandas.ParquetDataset** → `data/05_model_input/cb_X_test.parquet`
- `cb_y_train`: **kedro_datasets.pandas.ParquetDataset** → `data/05_model_input/cb_y_train.parquet`
- `cb_y_test`: **kedro_datasets.pandas.ParquetDataset** → `data/05_model_input/cb_y_test.parquet`
- `cb_cat_features`: **kedro_datasets.json.JSONDataset** → `data/05_model_input/cb_cat_features.json`
- `cb_model`: **pickle.PickleDataset** → `data/06_models/cb_model.pkl`
- `metrics_xgb`: **kedro_datasets.pandas.ParquetDataset** → `data/08_reporting/metrics_xgb.parquet`
- `metrics_cb`: **kedro_datasets.pandas.ParquetDataset** → `data/08_reporting/metrics_cb.parquet`
- `model_comparison_table`: **kedro_datasets.pandas.ParquetDataset** → `data/08_reporting/model_comparison.parquet`

## 🧠 Node Function Code (Top-Level Only)

### `preprocess_for_xgboost`
```python
def preprocess_for_xgboost(df: pd.DataFrame) -> pd.DataFrame:
    le_pillar = LabelEncoder()
    df["pillar_encoded"] = le_pillar.fit_transform(df["pillar"].astype(str))

    top_countries = df["countryCoor"].value_counts().nlargest(10).index
    df["country_clean"] = df["countryCoor"].where(df["countryCoor"].isin(top_countries), "Other")
    country_dummies = pd.get_dummies(df["country_clean"], prefix="country")

    df_xgb = pd.concat([
        df.drop(columns=["pillar", "countryCoor", "country_clean"]),
        country_dummies
    ], axis=1)

    return df_xgb
```

### `split_xgboost_data`
```python
def split_xgboost_data(df: pd.DataFrame):
    X = df.drop(columns=["startupDelay", "id"])
    y = df["startupDelay"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Wrap y Series as DataFrames
    return (
        X_train,
        X_test,
        y_train.to_frame(name="startupDelay"),
        y_test.to_frame(name="startupDelay"),
    )
```

### `train_xgboost_model`
```python
def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series, xgb_params: dict):
    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)
    return model
```

### `preprocess_for_catboost`
```python
def preprocess_for_catboost(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pillar"] = df["pillar"].astype(str)
    df["countryCoor"] = df["countryCoor"].astype(str)
    return df
```

### `split_catboost_data`
```python
def split_catboost_data(df: pd.DataFrame):
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=["startupDelay", "id"])
    y = df["startupDelay"]
    cat_features = ["pillar", "countryCoor"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Wrap y Series as DataFrames
    return (
        X_train,
        X_test,
        y_train.to_frame(name="startupDelay"),
        y_test.to_frame(name="startupDelay"),
        cat_features,
    )
```

### `train_catboost_model`
```python
def train_catboost_model(X_train, y_train, X_valid, y_valid, catboost_params, cat_features):
    model = CatBoostRegressor(**catboost_params)
    model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        cat_features=cat_features,
        verbose=0
    )
    return model
```

### `evaluate_cb_point_model`
```python
def evaluate_cb_point_model(cb_model: CatBoostRegressor, cb_X_test: pd.DataFrame, cb_y_test: pd.Series) -> pd.DataFrame:
    y_pred = cb_model.predict(cb_X_test)
    metrics = regression_metrics(cb_y_test, y_pred)
    return pd.DataFrame([metrics])
```

### `evaluate_xgb_model`
```python
def evaluate_xgb_model(xgb_model: XGBRegressor, xgb_X_test: pd.DataFrame, xgb_y_test: pd.Series) -> pd.DataFrame:
    y_pred = xgb_model.predict(xgb_X_test)
    metrics = regression_metrics(xgb_y_test, y_pred)
    return pd.DataFrame([metrics])
```

### `compare_model_metrics`
```python
def compare_model_metrics(metrics_xgb, metrics_cb):
    df_xgb = metrics_xgb.copy()
    df_cb_point = metrics_cb.copy()

    df_xgb["model"] = "xgboost"
    df_cb_point["model"] = "catboost_point"
    

    result = pd.concat([df_xgb, df_cb_point, df_cb_quantile], ignore_index=True)
    return result  # guaranteed to be a single DataFrame
```

### `preprocess`
```python
def preprocess(project_df, programme_df, org_df) -> pd.DataFrame:
    # --- Ensure all inputs are DataFrames ---
    project_df = pd.DataFrame(project_df)
    programme_df = pd.DataFrame(programme_df)
    org_df = pd.DataFrame(org_df)

    # --- Ensure expected numeric fields are properly typed ---
    numeric_cols = ["ecMaxContribution", "totalCost"]
    for col in numeric_cols:
        if col in project_df.columns:
            project_df[col] = pd.to_numeric(project_df[col], errors="coerce")
        else:
            project_df[col] = np.nan  # Create it to avoid downstream key errors

    # --- Parse dates safely ---
    for date_col in ["startDate", "endDate", "ecSignatureDate"]:
        if date_col in project_df.columns:
            project_df[date_col] = pd.to_datetime(project_df[date_col], errors="coerce")
        else:
            project_df[date_col] = pd.NaT

    # --- Derived features ---
    project_df["startupDelay"] = (project_df["startDate"] - project_df["ecSignatureDate"]).dt.days
    project_df["duration"] = (project_df["endDate"] - project_df["startDate"]).dt.days
    project_df["totalCostzero"] = project_df["totalCost"].fillna(0).eq(0).astype(int)

    # Safe ratio calculation
    def safe_ratio(row):
        try:
            if pd.notnull(row["ecMaxContribution"]) and pd.notnull(row["totalCost"]) and row["totalCost"] != 0:
                return float(row["ecMaxContribution"]) / float(row["totalCost"])
        except Exception:
            pass
        return None

    project_df["contRatio"] = project_df.apply(safe_ratio, axis=1)

    # --- Map legal basis to pillar ---
    mapping = {
        "HORIZON.1.1": "Pillar 1 - European Research Council (ERC)",
        "HORIZON.1.2": "Pillar 1 - Marie Sklodowska-Curie Actions (MSCA)",
        "HORIZON.1.3": "Pillar 1 - Research infrastructures",
        "HORIZON.2.1": "Pillar 2 - Health",
        "HORIZON.2.2": "Pillar 2 - Culture, creativity and inclusive society",
        "HORIZON.2.3": "Pillar 2 - Civil Security for Society",
        "HORIZON.2.4": "Pillar 2 - Digital, Industry and Space",
        "HORIZON.2.5": "Pillar 2 - Climate, Energy and Mobility",
        "HORIZON.2.6": "Pillar 2 - Food, Bioeconomy Natural Resources, Agriculture and Environment",
        "HORIZON.3.1": "Pillar 3 - The European Innovation Council (EIC)",
        "HORIZON.3.2": "Pillar 3 - European innovation ecosystems",
        "HORIZON.3.3": "Pillar 3 - Cross-cutting call topics",
        "EURATOM2027": "EURATOM2027",
        "EURATOM.1.1": "Improve and support nuclear safety...",
        "EURATOM.1.2": "Maintain and further develop expertise...",
        "EURATOM.1.3": "Foster the development of fusion energy...",
    }
    project_df["pillar"] = project_df.get("legalBasis", pd.Series(dtype=object)).map(mapping)

    # --- Merge with organization data ---
    org_df["projectID"] = org_df.get("projectID", pd.Series(dtype=object)).astype(str)
    project_df["id"] = project_df.get("id", pd.Series(dtype=object)).astype(str)

    # Coordinator country
    coor_map = org_df[org_df["role"] == "coordinator"][["projectID", "country"]]
    coor_map = coor_map.drop_duplicates("projectID").set_index("projectID")["country"]
    project_df["countryCoor"] = project_df["id"].map(coor_map)

    # Number of participating organizations
    number_org = org_df.groupby("projectID").size()
    project_df["numberOrg"] = project_df["id"].map(number_org).fillna(0).astype(int)

    # --- Drop unreasonable values ---
    project_df = project_df[project_df["startupDelay"] >= 0]

    # --- Final feature selection (drop any missing columns) ---
    expected_cols = [
        "id", "startupDelay", "totalCost", "totalCostzero",
        "ecMaxContribution", "duration", "contRatio",
        "pillar", "countryCoor", "numberOrg"
    ]
    selected = project_df.reindex(columns=[col for col in expected_cols if col in project_df.columns])

    return selected
```
