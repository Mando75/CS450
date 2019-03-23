from load_semeion import load_semeion
from load_letter import load_letter
from load_flare import load_flare

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier


def output(dataset_name, classifier_name, accuracy):
    print(f'|   {dataset_name}   |   {classifier_name}   |  {"{:.1f}%".format(accuracy)}   |')


def test_classifier(data, classifier, test_size=.33):
    training_data, testing_data, training_target, testing_target = train_test_split(
        data["data"], data["targets"], test_size=test_size, shuffle=True)
    classifier.fit(training_data, training_target)
    return accuracy_score(testing_target,
                          classifier.predict(testing_data)) * 100


##########################
# Classifiers
#########################
nn = MLPClassifier(hidden_layer_sizes=(85, ), max_iter=20000)
nb = GaussianNB()
knn = KNeighborsClassifier(n_neighbors=5)
bag = BaggingClassifier(base_estimator=knn, n_estimators=250)
adab = AdaBoostClassifier(n_estimators=90)
rf = RandomForestClassifier(n_estimators=70)

########################
# Datasets
#######################

if int(input("Run Semeion? (1/0): ")):
    semeion = load_semeion()
    print("|   Dataset   |   Classifier   |   Accuracy   |")
    print("--------------+----------------+--------------+")
    output("Semeion", "Neural Network", test_classifier(semeion, nn, .25))
    output("Semeion", "Naive Bayes", test_classifier(semeion, nb, .30))
    output("Semeion", "K Nearest Neighbors", test_classifier(semeion, knn, .25))
    output("Semeion", "Bagging", test_classifier(semeion, bag, .25))
    output("Semeion", "AdaBoost", test_classifier(semeion, adab, .30))
    output("Semeion", "Random Forest", test_classifier(semeion, rf, .25))

print("\n\n")

if int(input("Run Letter? (1/0): ")):
    letter = load_letter()
    print("|   Dataset   |   Classifier   |   Accuracy   |")
    print("--------------+----------------+--------------+")
    output("Letter", "Neural Network", test_classifier(letter, nn, .25))
    output("Letter", "Naive Bayes", test_classifier(letter, nb, .33))
    output("Letter", "K Nearest Neighbors", test_classifier(letter, knn, .25))
    output("Letter", "Bagging", test_classifier(letter, bag, .25))
    output("Letter", "AdaBoost", test_classifier(letter, adab, .25))
    output("Letter", "Random Forest", test_classifier(letter, rf, .25))

print("\n\n")

if int(input("Run Flare? (1/0): ")):
    flare = load_flare()
    print("|   Dataset   |   Classifier   |   Accuracy   |")
    print("--------------+----------------+--------------+")
    output("Flare", "Neural Network", test_classifier(flare, nn, .25))
    output("Flare", "Naive Bayes", test_classifier(flare, nb, .25))
    output("Flare", "K Nearest Neighbors", test_classifier(flare, knn, .25))
    output("Flare", "Bagging", test_classifier(flare, bag, .25))
    output("Flare", "AdaBoost", test_classifier(flare, adab, .25))
    output("Flare", "Random Forest", test_classifier(flare, rf, .25))
