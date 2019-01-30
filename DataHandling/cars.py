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
    test_label_encoding(cars_data)


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


def test_label_encoding(cars_data):
    fr_cars = label_encode_car(cars_data)
    targets = fr_cars["target"].values
    column_names = fr_cars.columns.values
    data = fr_cars[column_names[:-1]].values
    message = "Running label encoding"
    run_model(data, targets, message)


def label_encode_car(dataset):
    columns = [
        "buying", "maint", "doors", "persons", "lug_boot", "safety", "target"
    ]
    for col in columns:
        dataset[col] = dataset[col].astype("category")
        dataset[col] = pd.factorize(dataset[col])[0] + 1
    return dataset
