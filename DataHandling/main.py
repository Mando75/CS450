from DataHandling.cars import test_cars
from DataHandling.mpg import test_mpg
from DataHandling.stumath import test_stu_math
import pandas as pd


def main():
    # Tests the cars data with one hot and find/replace
    pd.set_option('display.width', 500)
    pd.set_option('display.max_columns', 30)
    test_cars()
    test_mpg()
    test_stu_math()


if __name__ == '__main__':
    main()
