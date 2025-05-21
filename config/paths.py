import os

# Define the root of the project (resolves regardless of where code is run from)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data folder
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")

# Helper function to generate paths to custom files
def data_path(filename):
    """Returns full path to a data file in the Data/ folder."""
    return os.path.join(DATA_DIR, filename)