"""Regression checks for preprocessing and checkpoint selection, without model training."""

import ast
import json
from pathlib import Path
import types
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


NOTEBOOK = json.loads((Path(__file__).resolve().parents[1] / "titanic.ipynb").read_text())


def cell_containing(text):
    for cell in NOTEBOOK["cells"]:
        source = cell["source"]
        source = "".join(source) if isinstance(source, list) else source
        if text in source:
            return source
    raise AssertionError(f"Notebook cell not found: {text}")


def passengers():
    return pd.DataFrame({
        "PassengerId": range(1, 9),
        "Survived": [0, 1, 0, 1, 1, 0, 0, 1],
        "Pclass": [1, 2, 3, 1, 2, 3, 1, 2],
        "Name": [f"Person, Mr. Example {i}" for i in range(8)],
        "Sex": ["male", "female"] * 4,
        "Age": [5, 15, 25, 40, 65, 30, 50, 22],
        "SibSp": [0, 1, 0, 2, 1, 0, 2, 0],
        "Parch": [0, 0, 1, 0, 1, 0, 0, 1],
        "Ticket": [str(i) for i in range(8)],
        "Fare": [8, 16, 24, 32, 40, 48, 56, 64],
        "Cabin": ["A1", "B1", "C1", "D1", "E1", "F1", "G1", None],
        "Embarked": ["S", "C", "Q", "S", "C", "Q", "S", "C"],
    })


def preprocess(frame):
    namespace = {"pd": pd, "np": np}
    nodes = [node for node in ast.parse(cell_containing("# Step 1: Handling missing values")).body
             if isinstance(node, ast.FunctionDef) and node.name == "preprocess_passengers"]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<notebook>", "exec"), namespace)
    return namespace["preprocess_passengers"](frame, passengers().drop(columns="Survived"))


def prepare_splits(frame, test_frame=None):
    namespace = {"pd": pd, "np": np, "train_test_split": train_test_split,
                 "train_df": frame.copy(),
                 "test_df": passengers().drop(columns="Survived") if test_frame is None else test_frame.copy()}
    exec(cell_containing("# Step 1: Handling missing values"), namespace)
    exec(cell_containing("# The validation rows were separated"), namespace)
    return namespace


class WorkflowTests(unittest.TestCase):
    def test_features_do_not_depend_on_survival_targets(self):
        frame = passengers()
        baseline_train, baseline_test = preprocess(frame)
        frame["Survived"] = 1 - frame["Survived"]
        changed_train, changed_test = preprocess(frame)
        pd.testing.assert_frame_equal(
            baseline_train.drop(columns="Survived"), changed_train.drop(columns="Survived")
        )
        pd.testing.assert_frame_equal(baseline_test, changed_test)

    def test_family_indicator_updates_the_original_frame(self):
        train, _ = preprocess(passengers())
        self.assertEqual(train["IsAlone"].tolist(), [1, 0, 0, 0, 0, 1, 0, 0])

    def test_validation_values_cannot_change_training_statistics_or_bin_edges(self):
        frame = pd.concat([passengers()] * 5, ignore_index=True)
        first = prepare_splits(frame)
        frame.loc[first["raw_validation"].index, ["Age", "Fare"]] = 1e6
        changed = prepare_splits(frame)
        pd.testing.assert_frame_equal(first["X_train"], changed["X_train"])
        pd.testing.assert_frame_equal(first["test_df"], changed["test_df"])
        self.assertFalse(set(first["X_train"].index) & set(first["X_val"].index))

    def test_test_batch_distribution_does_not_change_a_passengers_features(self):
        frame = pd.concat([passengers()] * 5, ignore_index=True)
        test = passengers().drop(columns="Survived")
        first = prepare_splits(frame, test)["test_df"].iloc[0]
        test.loc[1:, ["Age", "Fare"]] = 1e6
        changed = prepare_splits(frame, test)["test_df"].iloc[0]
        pd.testing.assert_series_equal(first, changed)

    def test_unknown_categories_missing_values_and_extreme_ages_are_handled(self):
        frame = pd.concat([passengers()] * 5, ignore_index=True)
        test = passengers().drop(columns="Survived")
        test.loc[0, ["Age", "Fare", "Embarked"]] = np.nan
        test.loc[1, ["Age", "Name", "Sex", "Cabin"]] = [120, "Unknown", "unknown", "Z9"]
        result = prepare_splits(frame, test)["test_df"]
        self.assertFalse(result.isna().any().any())
        self.assertEqual(result.loc[1, "AgeGroup"], 4)
        self.assertEqual(result.loc[1, "Sex"], -1)
        self.assertEqual(result.loc[1, "Deck"], -1)

    def test_constant_fares_do_not_break_quantile_binning(self):
        frame = pd.concat([passengers()] * 5, ignore_index=True)
        frame["Fare"] = 10.0
        result = prepare_splits(frame)
        self.assertEqual(result["X_train"]["FareBin"].unique().tolist(), [0])

    def test_fresh_run_uses_fitted_baseline_models(self):
        models = [object(), object(), object()]
        namespace = dict(LOAD_MODEL=False, HO_TUNING=False,
                         logreg=models[0], rf=models[1], xgb_model=models[2])
        exec(cell_containing("if LOAD_MODEL:"), namespace)
        self.assertIs(namespace["final_logreg"], models[0])
        self.assertIs(namespace["final_rf"], models[1])
        self.assertIs(namespace["final_xgb"], models[2])

    def test_tuned_models_are_not_replaced_by_baselines(self):
        tuned = object()
        namespace = dict(LOAD_MODEL=False, HO_TUNING=True,
                         final_logreg=tuned, final_rf=tuned, final_xgb=tuned)
        exec(cell_containing("if LOAD_MODEL:"), namespace)
        self.assertIs(namespace["final_xgb"], tuned)

    def test_explicit_checkpoint_loading_remains_available(self):
        namespace = dict(LOAD_MODEL=True, HO_TUNING=False,
                         joblib=types.SimpleNamespace(load=lambda path: path))
        exec(cell_containing("if LOAD_MODEL:"), namespace)
        self.assertTrue(namespace["final_logreg"].endswith("logistic_regression_model.pkl"))
        self.assertTrue(namespace["final_rf"].endswith("random_forest_model.pkl"))
        self.assertTrue(namespace["final_xgb"].endswith("xgboost_model.pkl"))


if __name__ == "__main__":
    unittest.main()
