from DataHandling.lib import run_model
import pandas as pd


def test_cars():
    print("#########################")
    print("  Testing Cars dataset")
    print("#########################")
    headers = [
        "buying", "maint", "doors", "persons", "lug_boot", "safety", "target"
    ]
    cars_data = pd.read_csv(
        "../datasets/car.data.csv", header=None, names=headers)
    test_one_hot(cars_data)
    test_find_replace(cars_data)


def test_one_hot(cars_data):
    hb_cars = one_hot_car(cars_data)
    clean_up_targets = {
        "target": {
            "unacc": 0,
            "acc": 1,
            "good": 2,
            "vgood": 3
        }
    }
    hb_cars.replace(clean_up_targets, inplace=True)
    targets = hb_cars["target"].values
    column_names = hb_cars.columns.values
    data = hb_cars[column_names[1:]].values
    message = "Running one hot"
    run_model(data, targets, message)


def one_hot_car(dataset):
    # Hot box data
    return pd.get_dummies(
        dataset,
        columns=["buying", "maint", "doors", "persons", "lug_boot", "safety"])


def test_find_replace(cars_data):
    fr_cars = find_replace_car(cars_data)
    targets = fr_cars["target"].values
    column_names = fr_cars.columns.values
    data = fr_cars[column_names[:-1]].values
    message = "Running find/replace"
    run_model(data, targets, message)


def find_replace_car(dataset):
    dataset["buying"] = dataset["buying"].astype("category")
    dataset["maint"] = dataset["maint"].astype("category")
    dataset["doors"] = dataset["doors"].astype("category")
    dataset["persons"] = dataset["persons"].astype("category")
    dataset["lug_boot"] = dataset["lug_boot"].astype("category")
    dataset["safety"] = dataset["safety"].astype("category")
    dataset["target"] = dataset["target"].astype("category")
    replacements = {
        "buying": {
            "vhigh": 4,
            "high": 3,
            "med": 2,
            "low": 1
        },
        "maint": {
            "vhigh": 4,
            "high": 3,
            "med": 2,
            "low": 1
        },
        "doors": {
            "5more": 5
        },
        "persons": {
            "more": 6
        },
        "lug_boot": {
            "small": 0,
            "med": 1,
            "big": 2
        },
        "safety": {
            "low": 0,
            "med": 1,
            "high": 2
        },
        "target": {
            "unacc": 0,
            "acc": 1,
            "good": 2,
            "vgood": 3
        }
    }
    dataset.replace(replacements, inplace=True)
    dataset["persons"] = dataset["persons"].astype("int")
    dataset["doors"] = dataset["doors"].astype("int")
    return dataset
