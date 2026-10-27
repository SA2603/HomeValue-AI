"""Train the HomeValue.ai model using the SAME Python that will serve it.
Run:  python train.py        (the app also calls this automatically when needed)
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from features import CITY_CENTERS, RAW_COLS, add_features

warnings.filterwarnings("ignore")
SEED = 42


def make_synthetic(n=12000, seed=SEED):
    r = np.random.default_rng(seed)
    cities = list(CITY_CENTERS.values())
    centre = [cities[i] for i in r.integers(0, len(cities), n)]
    lat = np.array([c[0] for c in centre]) + r.normal(0, 0.45, n)
    lon = np.array([c[1] for c in centre]) + r.normal(0, 0.45, n)
    inc = np.clip(r.lognormal(1.25, 0.5, n), 0.5, 15)
    age = r.integers(1, 53, n).astype(float)
    rooms = np.clip(r.normal(5.4, 1.4, n), 2, 10)
    beds = np.clip(rooms * r.normal(0.2, 0.03, n), 0.8, 3)
    pop = np.clip(r.lognormal(7.1, 0.7, n), 100, 10000)
    occ = np.clip(r.normal(3.0, 0.7, n), 1, 6)
    coast = np.exp(-np.sqrt((lat - 36) ** 2 + (lon + 120) ** 2) / 4)
    y = np.clip(0.45 * inc + 0.12 * rooms - 0.006 * age + 0.9 * coast + r.normal(0, 0.25, n), 0.15, 5.0)
    df = pd.DataFrame(dict(MedInc=inc, HouseAge=age, AveRooms=rooms, AveBedrms=beds,
                           Population=pop, AveOccup=occ, Latitude=lat, Longitude=lon))
    df["MedHouseVal"] = y
    return df


def load_data():
    try:
        from sklearn.datasets import fetch_california_housing
        return fetch_california_housing(as_frame=True).frame, "California Housing (1990 US census, scikit-learn)"
    except Exception:
        return make_synthetic(), "Synthetic demo data (same schema as California Housing)"


def train(n_iter=8, verbose=True):
    df, source = load_data()
    df = df.drop_duplicates().reset_index(drop=True)
    X, y = df[RAW_COLS], df["MedHouseVal"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)

    def build():
        pipe = Pipeline([("features", FunctionTransformer(add_features)),
                         ("model", HistGradientBoostingRegressor(random_state=SEED))])
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)

    grid = {
        "regressor__model__learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15],
        "regressor__model__max_iter": [200, 400, 600],
        "regressor__model__max_leaf_nodes": [15, 31, 63],
        "regressor__model__min_samples_leaf": [10, 20, 40],
        "regressor__model__l2_regularization": [0.0, 0.1, 1.0],
    }
    search = RandomizedSearchCV(build(), grid, n_iter=n_iter, cv=3, random_state=SEED,
                                scoring="neg_root_mean_squared_error", n_jobs=-1)
    search.fit(X_tr, y_tr)

    pred = search.best_estimator_.predict(X_te)
    ratio = (y_te / pred).clip(0.2, 5)
    meta = {
        "model_name": "Gradient Boosting",
        "r2": float(r2_score(y_te, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
        "mae": float(mean_absolute_error(y_te, pred)),
        "ratio_q10": float(np.quantile(ratio, 0.10)),
        "ratio_q90": float(np.quantile(ratio, 0.90)),
        "n_train": int(len(df)),
        "data_source": source,
        "sklearn_version": sklearn.__version__,
        "median_price": float(y.median() * 100_000),
    }
    final = search.best_estimator_.fit(X, y)
    joblib.dump(final, "homevalue_model.joblib")
    with open("model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    if verbose:
        print(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    train(n_iter=12)
