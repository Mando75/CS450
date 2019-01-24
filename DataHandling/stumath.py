import pandas as pd
import numpy as np
from DataHandling.lib import run_model


def test_stu_math():
    """
    Driver method to test the student math dataset
    :return: 
    """
    print("######################")
    print("Testing Math dataset")
    print("######################")
    math_data = pd.read_csv("../datasets/student-mat.csv", header=0, sep=";")
    # Move target column to front of data frame
    targets = math_data["G3"]
    math_data.drop(labels=["G3"], axis=1, inplace=True)
    math_data.insert(0, "G3", targets)
    test_one_hot(math_data)
    test_label_encoding(math_data)


def test_one_hot(dataset):
    """
    Tests the dataset with one hot encoding on category columns
    :param dataset: 
    :return: 
    """
    hb_math = one_hot_math(dataset)
    hb_math = replace_yes_no(hb_math)
    targets = hb_math["G3"].values
    col_names = hb_math.columns.values
    data = hb_math[col_names[1:]].values
    message = "running one hot on student math"
    run_model(data, targets, message, regression=True)


def one_hot_math(ds):
    """
    Creates one hot columns for the listed categories
    :param ds: 
    :return: 
    """
    return pd.get_dummies(
        ds,
        columns=[
            "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
            "reason", "guardian"
        ])


def replace_yes_no(ds):
    """
    Relabels the listed columns as 1 or 0 instead of yes or no
    :param ds: 
    :return: 
    """
    columns = [
        "schoolsup", "famsup", "paid", "activities", "nursery", "higher",
        "internet", "romantic"
    ]
    for col in columns:
        ds[col] = np.where(ds[col].str.contains("yes"), 1, 0)

    return ds


def test_label_encoding(ds):
    """
    Test's 
    :param ds: 
    :return: 
    """
    hb_math = label_encode_math(ds)
    targets = hb_math["G3"].values
    col_names = hb_math.columns.values
    data = hb_math[col_names[1:]].values
    message = "Running label encoding on math dataset"
    run_model(data, targets, message, regression=True)


def label_encode_math(ds):
    """
    Encodes the listed columns as integer labels starting at zero
    :param ds: 
    :return: 
    """
    hb_math = replace_yes_no(ds)
    columns = [
        "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
        "reason", "guardian"
    ]
    for col in columns:
        hb_math[col] = hb_math[col].astype("category")
        # Use factorize to create/know the label to use
        hb_math[col] = pd.factorize(hb_math[col])[0] + 1

    return hb_math
