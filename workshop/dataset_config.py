from pathlib import Path

# ⚙️ DATASET CONFIG — edit this to point to your data
DATASET_CONFIG = {
    "data_path": Path("../Airbnb"),        # root folder; subfolders = entities
    "entity_column": "city",               # what each subfolder represents
    "file_pattern": "listings.csv",        # which file to load per folder
    "keep_columns": None,                  # None = keep all columns
    "session_name": "analysis",            # used for DB, file, and thread naming
}