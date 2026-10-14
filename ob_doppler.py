from __future__ import annotations

import json
import math
import platform
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from config import ExperimentConfig

C_KMS = 299792.458


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_dataset(path: str | Path):
    """Load the private NPZ input used by the experiment scripts.

    Required arrays
    ---------------
    spectra     : shape (n_stars, n_wavelengths)
    wavelength  : shape (n_wavelengths,)
    labels      : broad O/B labels or spectral-type strings beginning with O/B

    Optional arrays
    ---------------
    star_names  : shape (n_stars,)
    subtypes    : shape (n_stars,)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        missing = {"spectra", "wavelength", "labels"} - set(data.files)
        if missing:
            raise ValueError(f"Missing required arrays in {path.name}: {sorted(missing)}")

        spectra = np.asarray(data["spectra"], dtype=np.float32)
        wavelength = np.asarray(data["wavelength"], dtype=np.float64)
        labels = np.asarray(data["labels"]).astype(str)
        star_names = (
            np.asarray(data["star_names"]).astype(str)
            if "star_names" in data.files
            else np.asarray([f"star_{i}" for i in range(len(spectra))], dtype=str)
        )
        subtypes = (
            np.asarray(data["subtypes"]).astype(str)
            if "subtypes" in data.files
            else np.asarray([""] * len(spectra), dtype=str)
        )

    if spectra.ndim != 2:
        raise ValueError("spectra must be a two-dimensional array.")
    if wavelength.ndim != 1 or spectra.shape[1] != len(wavelength):
        raise ValueError("wavelength must be one-dimensional and match spectra.shape[1].")
    if not (len(labels) == len(star_names) == len(subtypes) == len(spectra)):
        raise ValueError("labels, star_names and subtypes must match the number of spectra.")

    spectra = clean_spectra(spectra)
    y, keep = encode_ob_labels(labels)
    return (
        spectra[keep],
        wavelength,
        y,
        star_names[keep],
        subtypes[keep],
        labels[keep],
    )


def clean_spectra(spectra: np.ndarray, fill_value: float = 1.0) -> np.ndarray:
    """Replace non-finite samples by row-wise linear interpolation."""
    spectra = np.asarray(spectra, dtype=np.float32).copy()
    x = np.arange(spectra.shape[1])
    for i in range(len(spectra)):
        bad = ~np.isfinite(spectra[i])
        if not bad.any():
            continue
        good = ~bad
        if good.sum() >= 2:
            spectra[i, bad] = np.interp(x[bad], x[good], spectra[i, good])
        else:
            spectra[i, bad] = fill_value
    return spectra


def encode_ob_labels(labels: Iterable[str]):
    """Encode B=0 and O=1; non-O/B rows are excluded."""
    s = pd.Series(np.asarray(list(labels)).astype(str)).str.strip().str.upper()
    first = s.str[0]
    keep = first.isin(["O", "B"]).to_numpy()
    if keep.sum() == 0:
        raise ValueError("No O/B labels were found.")
    y = (first[keep] == "O").astype(int).to_numpy()
    return y, keep


def validate_manuscript_matrix(spectra, wavelength, y) -> dict:
    """Validate the numerical matrix described in the manuscript."""
    info = {
        "n_stars": int(len(spectra)),
        "n_B": int(np.sum(y == 0)),
        "n_O": int(np.sum(y == 1)),
        "n_wavelengths": int(spectra.shape[1]),
        "wavelength_min_A": float(wavelength[0]),
        "wavelength_max_A": float(wavelength[-1]),
        "median_step_A": float(np.median(np.diff(wavelength))),
    }
    expected = {
        "n_stars": 1455,
        "n_B": 1130,
        "n_O": 325,
        "n_wavelengths": 12199,
    }
    for key, value in expected.items():
        if info[key] != value:
            raise ValueError(f"{key}={info[key]} does not match manuscript value {value}.")
    if not np.isclose(info["wavelength_min_A"], 3800.25):
        raise ValueError("The wavelength grid must start at 3800.25 Angstrom.")
    if not np.isclose(info["wavelength_max_A"], 6849.75):
        raise ValueError("The wavelength grid must end at 6849.75 Angstrom.")
    if not np.isclose(info["median_step_A"], 0.25):
        raise ValueError("The wavelength sampling must be 0.25 Angstrom.")
    return info


def relativistic_doppler_factor(v_kms: float) -> float:
    beta = float(v_kms) / C_KMS
    if abs(beta) >= 1.0:
        raise ValueError("|v| must be smaller than the speed of light.")
    return math.sqrt((1.0 + beta) / (1.0 - beta))


def shift_linear(spectra, wavelength, v_kms: float, fill_value: float = 1.0):
    """Apply F_v(lambda)=F_0(lambda/D) and linearly resample to the fixed grid."""
    spectra = np.asarray(spectra, dtype=np.float32)
    one_dimensional = spectra.ndim == 1
    if one_dimensional:
        spectra = spectra[None, :]
    wavelength = np.asarray(wavelength, dtype=float)
    source = wavelength / relativistic_doppler_factor(v_kms)
    out = np.empty_like(spectra, dtype=np.float32)
    for i, row in enumerate(spectra):
        out[i] = np.interp(
            source,
            wavelength,
            row,
            left=fill_value,
            right=fill_value,
        ).astype(np.float32)
    return out[0] if one_dimensional else out


def shift_nearest(spectra, wavelength, v_kms: float, fill_value: float = 1.0):
    """Nearest-neighbor resampling used only in the interpolation sensitivity control."""
    spectra = np.asarray(spectra, dtype=np.float32)
    one_dimensional = spectra.ndim == 1
    if one_dimensional:
        spectra = spectra[None, :]
    wavelength = np.asarray(wavelength, dtype=float)
    source = wavelength / relativistic_doppler_factor(v_kms)
    valid = (source >= wavelength[0]) & (source <= wavelength[-1])

    right = np.searchsorted(wavelength, source, side="left")
    right = np.clip(right, 1, len(wavelength) - 1)
    left = right - 1
    use_right = np.abs(wavelength[right] - source) < np.abs(source - wavelength[left])
    nearest = np.where(use_right, right, left)

    out = np.full(spectra.shape, fill_value, dtype=np.float32)
    out[:, valid] = spectra[:, nearest[valid]]
    return out[0] if one_dimensional else out


def augment_once(spectra, y, wavelength, velocities, operator=shift_linear, fill_value=1.0):
    """Append one shifted copy of each training spectrum."""
    spectra = np.asarray(spectra, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    velocities = np.asarray(velocities, dtype=float)
    if len(velocities) != len(spectra):
        raise ValueError("velocities must contain one value per training spectrum.")

    shifted = np.empty_like(spectra, dtype=np.float32)
    for i, velocity in enumerate(velocities):
        shifted[i] = operator(spectra[i], wavelength, float(velocity), fill_value)
    return np.vstack([spectra, shifted]), np.concatenate([y, y])


def build_models(seed: int, n_jobs: int = -1, include_decision_tree: bool = True):
    """Build the fixed classifier configurations used in the study."""
    models = {}

    if include_decision_tree:
        models["DecisionTree"] = DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=2,
            min_samples_split=3,
            random_state=seed,
        )

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        max_features="sqrt",
        min_samples_split=4,
        random_state=seed,
        n_jobs=n_jobs,
    )

    models["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=50,
        max_depth=6,
        max_features="sqrt",
        min_samples_split=4,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=n_jobs,
    )

    models["HistGradientBoosting"] = HistGradientBoostingClassifier(
        max_depth=9,
        max_bins=20,
        learning_rate=0.3566666666666667,
        max_iter=100,
        random_state=seed,
    )

    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError("lightgbm is required. Install packages from requirements.txt.") from exc

    models["LightGBM"] = LGBMClassifier(
        n_estimators=200,
        max_depth=15,
        num_leaves=140,
        min_child_samples=10,
        learning_rate=0.2826666666666667,
        colsample_bytree=0.96,
        reg_alpha=0.8,
        reg_lambda=6.0,
        subsample=0.56,
        subsample_freq=0,
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=-1,
    )

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is required. Install packages from requirements.txt.") from exc

    models["XGBoost"] = XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.042222222222222223,
        subsample=0.9,
        reg_lambda=0.1,
        reg_alpha=0.01,
        min_child_weight=7,
        gamma=0.1,
        colsample_bytree=1.0,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=n_jobs,
    )

    return {
        name: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", model),
            ]
        )
        for name, model in models.items()
    }


def classification_metrics(y_true, y_pred, p_O) -> dict:
    """Return the metrics reported in the study; O is the positive class."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, p_O)),
        "brier": float(brier_score_loss(y_true, p_O)),
        "recall_B": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_O": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def corrected_repeated_split_test(differences, test_fraction=0.25, train_fraction=0.75) -> dict:
    """Nadeau-Bengio corrected repeated-split t-test."""
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        raise ValueError("At least two paired split differences are required.")
    mean = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    factor = float(np.sqrt(1.0 / len(d) + test_fraction / train_fraction))
    se = sd * factor
    if se == 0.0:
        t_stat = np.inf if mean != 0.0 else 0.0
        p_value = 0.0 if mean != 0.0 else 1.0
    else:
        t_stat = mean / se
        p_value = float(2.0 * student_t.sf(abs(t_stat), df=len(d) - 1))
    return {
        "n": int(len(d)),
        "mean": mean,
        "sd": sd,
        "correction_factor": factor,
        "t": float(t_stat),
        "df": int(len(d) - 1),
        "p": p_value,
    }


def runtime_versions() -> dict:
    import lightgbm
    import scipy
    import sklearn
    import xgboost

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__,
    }


def write_run_metadata(output_dir: str | Path, cfg: ExperimentConfig, dataset_info: dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "configuration": cfg.to_dict(),
        "dataset": dataset_info,
        "software": runtime_versions(),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
