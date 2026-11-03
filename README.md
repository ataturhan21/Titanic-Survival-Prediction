# Titanic Survival Prediction

An exploratory machine-learning notebook for Kaggle's Titanic competition, covering feature engineering, model comparison, optional hyperparameter tuning, stacking, and prediction explanations.

**[Open the notebook](titanic.ipynb)** · **[Kaggle version](https://www.kaggle.com/code/ataturhan/titanic-predicting-survivors)** · **[Dependencies](requirements.txt)** · **[MIT license](LICENSE)**

## What's implemented

- Passenger-data exploration and features such as age groups, fare bins, family size, titles, and deck.
- Logistic regression, random forest, multilayer perceptron, and XGBoost baselines.
- Optional grid searches and a stacking classifier.
- Model explanations using coefficients, feature importance, permutation importance, and SHAP.
- A `submission.csv` containing `PassengerId` and `Survived`.

## Saved results

The committed outputs come from an earlier version of the notebook that still used the target-derived `FamilySurvival` feature, which has since been removed (see below). Read these numbers as a historical record that is likely optimistic, not as scores of the current code.

| Model | Saved validation accuracy (179 rows) |
| --- | --- |
| Logistic regression | 0.8045 |
| Random forest | 0.8268 |
| Multilayer perceptron | 0.7933 |
| XGBoost | 0.8212 |

The cross-validation, tuning, stacking and submission cells have no saved output.

## Run on Kaggle

1. Create a notebook and attach the [Titanic competition data](https://www.kaggle.com/competitions/titanic/data).
2. Import `titanic.ipynb`.
3. Confirm that the competition files are at these paths (the notebook uses them whenever `/kaggle/input/titanic` exists):

   ```text
   /kaggle/input/titanic/train.csv
   /kaggle/input/titanic/test.csv
   ```

4. Leave the default settings for a fresh run:

   ```python
   HO_TUNING = False
   LOAD_MODEL = False
   ```

5. Run the cells in order. Stacking will use the baseline models fitted earlier in the notebook.
6. Download `submission.csv` from the notebook's working directory.

Set `HO_TUNING=True` to run the grid searches and save tuned models under `MODEL_DIR`: `/kaggle/working/models/` on Kaggle, or `models/` next to the notebook elsewhere. Set `LOAD_MODEL=True` only when the expected logistic-regression, random-forest, and XGBoost checkpoint files already exist there.

## Run locally

```bash
git clone https://github.com/ataturhan21/Titanic-Survival-Prediction.git
cd Titanic-Survival-Prediction
python -m venv .venv
```

Activate the environment, then run:

```bash
python -m pip install -r requirements.txt notebook
jupyter notebook titanic.ipynb
```

Download `train.csv` and `test.csv` from the competition and put them in a `data/` folder next to the notebook. When the Kaggle folders do not exist, the notebook reads from `data/` and saves checkpoints to `models/`, so no path edits are needed. Datasets and checkpoints are not included in this repository; `data/`, `models/` and `submission.csv` are ignored by Git.

## Evaluation and reproducibility

The target-derived `FamilySurvival` feature has been removed because it used survival labels before the validation split. Family-size features that do not use the target remain available.

The raw data is split before preprocessing. Group medians and fare-bin boundaries are learned from the training rows and reused for validation and test rows; target-independent category mappings stay consistent across all splits. Missing/unknown categories, extreme ages, and constant fare distributions have explicit handling.

Optional grid search and stacking still receive preprocessed training features. Their inner cross-validation should place learned preprocessing inside each fold before its scores are used as independent evidence. The separate held-out validation rows do not determine preprocessing statistics. Saved outputs are records of an earlier experiment, not verified scores for the current code.

Saved notebook metadata records Python 3.10.13. Dependencies are unpinned; record the actual environment and rerun the workflow before reporting results.

## Focused regression checks

```bash
python -m pip install numpy pandas scikit-learn
python -m unittest discover -s tests -v
```

These checks verify held-out data isolation, consistent train/test features, unknown/missing values, constant-fare handling, target-independent family features, baseline/tuned/checkpoint selection, and the Kaggle-or-local data and model folders. They use NumPy, pandas, and scikit-learn; they do not train the full model stack or rerun SHAP. A read-only GitHub Actions workflow runs the same focused checks on Ubuntu and Windows.

## License

[MIT](LICENSE).
