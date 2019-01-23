from cars import test_cars


def main():
    print("######################")
    print(" TESTING WITH MY KNN  ")
    print("######################")
    test_cars(sklearn=False)

    print("###########################")
    print(" TESTING WITH SKLEARN KNN  ")
    print("###########################")
    test_cars(sklearn=True)


if __name__ == '__main__':
    main()
