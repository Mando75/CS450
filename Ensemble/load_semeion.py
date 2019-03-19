import pandas as pd


def load_semeion():
    sem_data = pd.read_csv(
        "../datasets/semeion.data.csv", header=None, delimiter=' ')
    sem_data.drop(
        sem_data.columns[len(sem_data.columns) - 1],
        axis=1,
        inplace=True,
    )
    struct = {}
    struct["targets"] = sem_data.apply(lambda row: number_mapping(row), axis=1)
    struct["data"] = sem_data.drop([i for i in range(256, 266)], axis=1)
    return struct


def number_mapping(row):
    if row[256] == 1:
        return 0
    if row[257] == 1:
        return 1
    if row[258] == 1:
        return 2
    if row[259] == 1:
        return 3
    if row[260] == 1:
        return 4
    if row[261] == 1:
        return 5
    if row[262] == 1:
        return 6
    if row[263] == 1:
        return 7
    if row[264] == 1:
        return 8
    if row[265] == 1:
        return 9
