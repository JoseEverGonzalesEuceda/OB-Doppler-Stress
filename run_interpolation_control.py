from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import ExperimentConfig
from ob_doppler import (
    augment_once,
    build_models,
    classification_metrics,
    load_dataset,
    set_seed,
    shift_linear,
    shift_nearest,
    validate_manuscript_matrix,
    write_run_metadata,
)

CONTROL_MODELS = ("ExtraTrees", "HistGradientBoosting", "XGBoost")


def run(data_path: Path, output_dir: Path, cfg: ExperimentConfig, validate: bool = True):
    spectra, wavelength, y, star_names, _, _ = load_dataset(data_path)
    if validate:
        dataset_info = validate_manuscript_matrix(spectra, wavelength, y)
    else:
        dataset_info = {
            "n_stars": int(len(spectra)),
            "n_B": int(np.sum(y == 0)),
            "n_O": int(np.sum(y == 1)),
            "n_wavelengths": int(spectra.shape[1]),
            "wavelength_min_A": float(wavelength[0]),
            "wavelength_max_A": float(wavelength[-1]),
            "median_step_A": float(np.median(np.diff(wavelength))),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []

    for repeat, seed in enumerate(cfg.seeds, start=1):
        set_seed(seed)
        indices = np.arange(len(spectra))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=cfg.test_size,
            stratify=y,
            random_state=seed,
        )
        X_train, X_test = spectra[train_idx], spectra[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Independent, reproducible augmentation draw used by the control run.
        rng = np.random.default_rng(seed * 1000 + 777)
        train_velocities = rng.uniform(
            -cfg.augmentation_vmax_kms,
            cfg.augmentation_vmax_kms,
            size=len(X_train),
        )

        X_nearest, y_nearest = augment_once(
            X_train,
            y_train,
            wavelength,
            train_velocities,
            operator=shift_nearest,
            fill_value=cfg.fill_value,
        )
        X_linear, y_linear = augment_once(
            X_train,
            y_train,
            wavelength,
            train_velocities,
            operator=shift_linear,
            fill_value=cfg.fill_value,
        )

        training_sets = {
            "raw": (X_train, y_train),
            "raw_dup": (np.vstack([X_train, X_train]), np.concatenate([y_train, y_train])),
            "nearest_aug": (X_nearest, y_nearest),
            "linear_aug": (X_linear, y_linear),
        }

        fitted = {}
        for strategy, (X_fit, y_fit) in training_sets.items():
            models = build_models(seed, cfg.n_jobs, include_decision_tree=False)
            models = {name: model for name, model in models.items() if name in CONTROL_MODELS}
            fitted[strategy] = {}
            for model_name, model in models.items():
                print(f"seed={seed:3d} strategy={strategy:11s} model={model_name}")
                model.fit(X_fit, y_fit)
                fitted[strategy][model_name] = model

        test_operators = {
            "linear_test": shift_linear,
            "nearest_test": shift_nearest,
        }
        for test_operator_name, operator in test_operators.items():
            for velocity in cfg.control_velocities_kms:
                X_eval = (
                    X_test
                    if velocity == 0
                    else operator(X_test, wavelength, velocity, cfg.fill_value)
                )
                for strategy, model_dict in fitted.items():
                    for model_name, model in model_dict.items():
                        y_pred = model.predict(X_eval)
                        p_O = model.predict_proba(X_eval)[:, 1]
                        row = {
                            "repeat": repeat,
                            "seed": seed,
                            "test_operator": test_operator_name,
                            "train_strategy": strategy,
                            "model": model_name,
                            "velocity_kms": float(velocity),
                            "abs_velocity_kms": abs(float(velocity)),
                        }
                        row.update(classification_metrics(y_test, y_pred, p_O))
                        metric_rows.append(row)

    pd.DataFrame(metric_rows).to_csv(output_dir / "control_metrics.csv", index=False)
    write_run_metadata(output_dir, cfg, dataset_info)


def main():
    parser = argparse.ArgumentParser(
        description="Run the matched-size and interpolation sensitivity control."
    )
    parser.add_argument("--data", required=True, type=Path, help="Private dataset.npz file")
    parser.add_argument("--output", default=Path("results/control"), type=Path)
    parser.add_argument(
        "--skip-manuscript-validation",
        action="store_true",
        help="Allow a matrix that differs from the published 1455-star input.",
    )
    args = parser.parse_args()

    run(
        args.data,
        args.output,
        ExperimentConfig(),
        validate=not args.skip_manuscript_validation,
    )


if __name__ == "__main__":
    main()
