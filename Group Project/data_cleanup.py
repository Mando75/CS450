import pandas as pd

core_columns = [
    "CS 124", "CS 165", "CS 235", "CS 124 Average", "CS 165 Average",
    "CS 235 Average", "Core Average"
]


def load_data(use_means=True):
    data = pd.read_csv("../datasets/linkedenrollments.csv")
    if use_means:
        data = impute_means(data)
    else:
        data = impute_zero(data)
    data = clean_up_na_scores(data)
    data = drop_helper_columns(data)
    data = encode_courses(data)
    data = one_hot_columns(data)
    print("Data")
    print(data.head())
    split = split_data_targets(data)
    return split


def clean_up_na_scores(data):
    # Checked the dataset, this would be the correct value
    # in all cases
    data["finalGrade"].fillna('F', inplace=True)
    data["currentScore"].fillna(data["finalScore"], inplace=True)
    data["currentGrade"].fillna(data["finalGrade"], inplace=True)
    return data


def impute_means(data):
    columns = ["finalScore"] + core_columns
    for col in columns:
        data[col].fillna(data[col].mean(), inplace=True)
    return data


def impute_zero(data):
    columns = ["currentScore", "finalScore"] + core_columns
    for col in columns:
        data[col].fillna(0, inplace=True)
    return data


def drop_helper_columns(
        data, columns=["teacherName", "courseId", "userId"] + core_columns):
    return data.drop(columns=columns)


def encode_courses(data):
    columns = ["currentGrade", "finalGrade", "courseCode"]
    for col in columns:
        data[col] = data[col].astype("category")
        data[col] = pd.factorize(data[col])[0] + 1
    return data


def one_hot_columns(data):
    return pd.get_dummies(data, columns=["teacherId"])


def split_data_targets(data):
    targets = data["currentScore"]
    data = data.drop(columns=["currentScore"])
    struct = {}
    struct["targets"] = targets
    struct["data"] = data
    return struct
