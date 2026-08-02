"""Shared configuration for the tourism purchase-prediction project."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "tourism.csv"
DATA_MANIFEST_PATH = PROJECT_ROOT / "data" / "dataset_manifest.json"
SPLIT_DIR = PROJECT_ROOT / "artifacts" / "splits"
REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports"
DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"

TARGET = "ProdTaken"
RANDOM_STATE = 42
TEST_SIZE = 0.20

EXPECTED_COLUMNS = [
    "Unnamed: 0",
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]

IDENTIFIER_COLUMNS = ["Unnamed: 0", "CustomerID"]

CATEGORICAL_FEATURES = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation",
]

NUMERIC_FEATURES = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

