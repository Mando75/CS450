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
    # test_one_hot(mpg_data)
    test_find_replace_mpg(mpg_data)


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


def test_find_replace_mpg(mpg_data):
    hb_mpg = find_replace_mpg(mpg_data)
    targets = hb_mpg["mpg"].values
    col_names = hb_mpg.columns.values
    data = hb_mpg[col_names[1:]].values
    message = "Running Find and Replace"
    run_model(data, targets, message, regression=True)


def find_replace_mpg(dataset):
    cats = get_car_makes(dataset["car name"])
    dataset["car name"] = dataset["car name"].astype("str")
    for cat in cats:
        dataset["car name"].replace(
            r'.*' + re.escape(cat[0]) + r'.*',
            str(cat[1]),
            regex=True,
            inplace=True)

    dataset["model year"] = dataset["model year"].astype("category")
    dataset["model year"] = pd.factorize(dataset["model year"])[0] + 1
    dataset["cylinders"] = dataset["cylinders"].astype("category")
    dataset["cylinders"] = pd.factorize(dataset["cylinders"])[0] + 1
    dataset["origin"] = dataset["origin"].astype("category")
    dataset["origin"] = pd.factorize(dataset["origin"])[0] + 1
    return dataset


def get_car_makes(data):
    cats = set()
    for car in data:
        cats.add((car.split(" ", 1)[0]))
    return [(cat, index) for index, cat in enumerate(cats)]
