import pytest
import pandas as pd

from pathlib import Path
from starter.model.model import train_model, compute_model_metrics, inference
from starter.data.data import process_data
from starter.data.cleaning import clean_data


@pytest.fixture(scope="session")
def load_data():
    data_path = Path(__file__).parent / "test_data" / "census.csv"
    data = pd.read_csv(data_path, nrows=99)
    return data


@pytest.fixture(scope="session")
def cleaned_path():
    return Path(__file__).parent / "test_data" / "census_cleaned.csv"


def test_load_data(load_data):
    assert load_data.shape == (99, 15)


def test_cleaning(load_data, cleaned_path):
    cleaned_data, cat_cols, num_cols = clean_data(load_data, cleaned_path, "salary")
    assert cleaned_data.shape == (99, 15)
    assert cleaned_data.isna().sum().sum() == 0
    assert len(cat_cols) == 8
    assert len(num_cols) == 6


def test_process_data(load_data, cleaned_path):
    cleaned_data, cat_cols, num_cols = clean_data(
        load_data, cleaned_path, "salary", test=True
    )
    X_train, y_train, encoder, lb = process_data(
        cleaned_data, categorical_features=cat_cols, label="salary", training=True
    )
    assert X_train.shape == (99, 65)
    assert y_train.shape == (99,)
    assert encoder is not None
    assert lb is not None


def test_train_model(load_data, cleaned_path):
    cleaned_data, cat_cols, num_cols = clean_data(
        load_data, cleaned_path, "salary", test=True
    )
    X_train, y_train, encoder, lb = process_data(
        cleaned_data, categorical_features=cat_cols, label="salary", training=True
    )
    model = train_model(X_train, y_train)
    assert model is not None


def test_compute_model_metrics(load_data, cleaned_path):
    cleaned_data, cat_cols, num_cols = clean_data(
        load_data, cleaned_path, "salary", test=True
    )
    X_train, y_train, encoder, lb = process_data(
        cleaned_data, categorical_features=cat_cols, label="salary", training=True
    )
    model = train_model(X_train, y_train)
    X_test, y_test, _, _ = process_data(
        cleaned_data,
        categorical_features=cat_cols,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    y_pred = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
    assert precision is not None
    assert recall is not None
    assert fbeta is not None


def test_inference(load_data, cleaned_path):
    cleaned_data, cat_cols, num_cols = clean_data(
        load_data, cleaned_path, "salary", test=True
    )
    X_train, y_train, encoder, lb = process_data(
        cleaned_data, categorical_features=cat_cols, label="salary", training=True
    )
    model = train_model(X_train, y_train)
    X_test, y_test, _, _ = process_data(
        cleaned_data,
        categorical_features=cat_cols,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    y_pred = inference(model, X_test)
    assert y_pred is not None
