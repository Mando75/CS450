import pandas as pd
from numpy import NaN
import re
from DataHandling.lib import run_model


def test_mpg():
    print("#########################")
    print("  Testing MPG dataset")
    print("#########################")
    headers = [
        "mpg", "cylinders", "displacement", "horsepower", "weight",
        "acceleration", "model year", "origin", "car name"
    ]
    mpg_data = pd.read_table(
        "../datasets/auto-mpg.data.csv", sep='\s+', header=None, names=headers)
    mpg_data = handle_empties(mpg_data)
    test_one_hot(mpg_data)
    test_combo_mpg(mpg_data)


def handle_empties(dataset):
    """
    Fills in ? values in the horsepower column with the mean
    of the column
    :param dataset: 
    :return: 
    """
    dataset["horsepower"] = dataset["horsepower"].replace("?", NaN)
    dataset["horsepower"] = dataset["horsepower"].astype("float")
    dataset["horsepower"].fillna(dataset["horsepower"].mean(), inplace=True)
    return dataset


def test_one_hot(mpg_data):
    hb_mpg = one_hot_mpg(mpg_data)
    targets = hb_mpg["mpg"].values
    col_names = hb_mpg.columns.values
    data = hb_mpg[col_names[1:]].values
    message = "Running one hot"
    run_model(data, targets, message, regression=True)


def one_hot_mpg(mpg_data):
    return pd.get_dummies(
        mpg_data, columns=["car name", "model year", "cylinders", "origin"])


def test_combo_mpg(mpg_data):
    hb_mpg = combo_manip_mpg(mpg_data)
    targets = hb_mpg["mpg"].values
    col_names = hb_mpg.columns.values
    data = hb_mpg[col_names[1:]].values
    message = "Running Combo Binary and Label encodings"
    run_model(data, targets, message, regression=True)


def combo_manip_mpg(dataset):
    # Create a binary encoding for car names
    # based on make (first word in car name)
    cats = get_car_makes(dataset["car name"])
    dataset["car name"] = dataset["car name"].astype("str")
    for cat in cats:
        dataset["car name"].replace(
            r'.*' + re.escape(cat[0]) + r'.*',
            cat[1],
            regex=True,
            inplace=True)

    # Label encoding for other category columns
    columns = ["model year", "cylinders", "origin"]
    for col in columns:
        dataset[col] = dataset[col].astype("category")
        dataset[col] = pd.factorize(dataset[col])[0] + 1
    return dataset


def get_car_makes(data):
    """
    Get's unique car makes from the dataset
    :param data: 
    :return: 
    """
    cats = set()
    for car in data:
        cats.add((car.split(" ", 1)[0]))
    return [(cat, index) for index, cat in enumerate(cats)]
