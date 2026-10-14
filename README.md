# OB-Doppler-Stress

Code accompanying the study **“Stress-testing Tree-based O/B Stellar Spectral Classification under Controlled Doppler Shifts.”**

The repository contains only the software needed to reproduce the computational experiments. It does not contain observational spectra, labels, derived prediction tables, or result files.

## Files in this repository

- `config.py` — fixed experiment configuration.
- `ob_doppler.py` — data loading, Doppler operators, classifiers, metrics, and statistical utilities.
- `run_main_experiment.py` — Raw versus Raw+Aug experiment.
- `run_interpolation_control.py` — matched-size and interpolation sensitivity control.
- `summarize_results.py` — corrected repeated-split statistics, bootstrap summary, and subtype-group summaries.
- `requirements.txt` — Python dependencies used for the experiments.
- `.gitignore` — prevents private spectra and generated results from being committed.
- `CITATION.cff` — software citation metadata.
- `LICENSE` — BSD 3-Clause license for the code.

## Data access

No observational data are distributed with this repository or with its Zenodo software archive.

The spectra used in the study originate from the **IACOB spectroscopic database**. Access to the observational material must be obtained directly from the IACOB project and remains subject to its own access, acknowledgement, citation, and redistribution conditions:

https://research.iac.es/proyecto/iacob/iacobcat/

The principal IACOB reference used by this project is:

> Simón-Díaz, S., Castro, N., Garcia, M., Herrero, A., & Markova, N. (2011). The IACOB spectroscopic database of Northern Galactic OB stars. *Bulletin de la Société Royale des Sciences de Liège*, 80, 514–518.

Suggested acknowledgement:

> This research has made use of the IACOB spectroscopic database, operated at the Instituto de Astrofísica de Canarias (Tenerife, Spain).

The BSD 3-Clause license in this repository applies to the software only. It does not grant rights to redistribute IACOB spectra or other third-party observational material.

## Numerical input used in the study

After selection of O- and B-type objects, the numerical matrix analyzed in the study contains:

- 1455 stars: 1130 B and 325 O;
- 12,199 flux samples per spectrum;
- wavelength range 3800.25–6849.75 Å;
- sampling of 0.25 Å;
- spectra previously homogenized to an equivalent resolving power of `R = 4000`.

The scripts treat the original spectrum as the observed-frame reference at `v = 0`; it is not assumed to be corrected to the stellar rest frame.

## Private input format

The scripts read one local `.npz` file. This file is private and must not be committed to GitHub or uploaded to Zenodo.

Required arrays:

```text
spectra      shape (n_stars, n_wavelengths)
wavelength   shape (n_wavelengths,)
labels       shape (n_stars,)
```

Optional arrays:

```text
star_names   shape (n_stars,)
subtypes     shape (n_stars,)
```

`labels` may contain `O`/`B` or detailed spectral strings beginning with `O` or `B`. Internally, the code uses `B = 0` and `O = 1`.

For an existing Python session containing `df`, `y`, and `wave`, a private input file can be created with:

```python
import numpy as np

np.savez_compressed(
    r"C:\private_iacob\dataset.npz",
    spectra=df.to_numpy(dtype=np.float32),
    wavelength=np.asarray(wave, dtype=np.float64),
    labels=np.asarray(y).astype(str),
    star_names=df.index.astype(str).to_numpy(),
)
```

If subtype labels are available, add `subtypes=np.asarray(subtypes).astype(str)` to the same call.

## Installation

Python 3.12 or 3.13 is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install the recorded dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Main experiment

The main experiment uses ten stratified 75/25 hold-outs with seeds:

```text
11, 23, 37, 41, 53, 67, 79, 83, 97, 101
```

For Raw+Aug, one additional shifted copy is generated for each training spectrum. Training velocities are drawn uniformly from `[-100,+100] km/s`. Evaluation is performed from `-200` to `+200 km/s` in 20 km/s steps.

Run:

```bash
python run_main_experiment.py --data "C:\private_iacob\dataset.npz"
```

The default local output directory is `results/main/`. It is created automatically and ignored by Git.

## Matched-size and interpolation control

The control uses Extra Trees, HistGradientBoosting, and XGBoost with four training conditions:

- `raw` — original training set;
- `raw_dup` — exact duplicate of every training row;
- `nearest_aug` — one shifted copy using nearest-neighbor resampling;
- `linear_aug` — one shifted copy using linear interpolation.

Both linear and nearest-neighbor test operators are evaluated at `0`, `±100`, and `±200 km/s`.

Run:

```bash
python run_interpolation_control.py --data "C:\private_iacob\dataset.npz"
```

The default local output directory is `results/control/`.

## Statistical summary

After both experiments finish:

```bash
python summarize_results.py
```

The script reports:

- sign-averaged Raw+Aug − Raw balanced-accuracy differences at `|v| = 200 km/s`;
- the Nadeau–Bengio corrected repeated-split statistic;
- the matched-size/interpolation control summary;
- the XGBoost star-cluster bootstrap at `|v| = 200 km/s`;
- subtype-group error summaries when subtype information is present in the private input.

All generated CSV files remain local under `results/` and are excluded from version control.

## Doppler convention

For velocity `v`,

```text
D = sqrt((1 + beta)/(1 - beta)),    beta = v/c
F_v(lambda) = F_0(lambda / D)
```

Positive velocity shifts spectral features toward longer wavelengths. Linear interpolation is used in the main experiment, with values outside the common grid filled by normalized continuum flux `1.0`.

## Classifier configurations

The classifier hyperparameters are fixed to the configurations inherited from the preceding classification study. They are not re-optimized for the binary O/B task or for the Doppler perturbation experiments. `LightGBM subsample=0.56` is retained from that configuration, while `subsample_freq=0` leaves row subsampling inactive.

## License

The code is released under the BSD 3-Clause License. The license applies only to the software in this repository.
