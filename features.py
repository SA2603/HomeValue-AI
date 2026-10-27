"""Shared feature engineering for HomeValue.ai (used by the notebook AND the web app)."""
import numpy as np
import pandas as pd

RAW_COLS = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"]

CITY_CENTERS = {
    "Los Angeles":   (34.05, -118.24),
    "San Francisco": (37.77, -122.42),
    "San Diego":     (32.72, -117.16),
    "San Jose":      (37.34, -121.89),
    "Sacramento":    (38.58, -121.49),
}


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Take the 8 raw columns, return the engineered feature table."""
    X = pd.DataFrame(X, columns=RAW_COLS).copy()
    X["BedroomRatio"] = X["AveBedrms"] / X["AveRooms"]
    X["RoomsPerPerson"] = X["AveRooms"] / X["AveOccup"]
    X["LogPopulation"] = np.log1p(X["Population"])
    dist_cols = []
    for city, (lat, lon) in CITY_CENTERS.items():
        col = f"Dist_{city.replace(' ', '')}"
        X[col] = np.sqrt((X["Latitude"] - lat) ** 2 + (X["Longitude"] - lon) ** 2)
        dist_cols.append(col)
    X["DistNearestCity"] = X[dist_cols].min(axis=1)
    return X
