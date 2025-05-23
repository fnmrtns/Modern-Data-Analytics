import pandas as pd
import numpy as np

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
