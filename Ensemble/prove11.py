from load_semeion import load_semeion
from load_letter import load_letter
from load_flare import load_flare

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def printResults(dataset_name, classifier_name, accuracy):
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
nn = MLPClassifier(hidden_layer_sizes=(75, ), max_iter=20000)
nb = GaussianNB()
knn = KNeighborsClassifier(n_neighbors=7)
########################
# Datasets
#######################
semeion = load_semeion()
letter = load_letter()
flare = load_flare()

print("|   Dataset   |   Classifier   |   Accuracy   |")
print("--------------+----------------+--------------+")
printResults("Semeion", "Neural Network", test_classifier(semeion, nn, .25))
printResults("Semeion", "Naive Bayes", test_classifier(semeion, nb, .25))
printResults("Semeion", "K Nearest Neighbors", test_classifier(semeion, knn, .25))

print("\n\n")

print("|   Dataset   |   Classifier   |   Accuracy   |")
print("--------------+----------------+--------------+")
printResults("Letter", "Neural Network", test_classifier(letter, nn, .25))
printResults("Letter", "Naive Bayes", test_classifier(letter, nb, .25))
printResults("Letter", "K Nearest Neighbors", test_classifier(letter, knn, .25))

print("\n\n")

print("|   Dataset   |   Classifier   |   Accuracy   |")
print("--------------+----------------+--------------+")
printResults("Flare", "Neural Network", test_classifier(flare, nn, .25))
printResults("Flare", "Naive Bayes", test_classifier(flare, nb, .25))
printResults("Flare", "K Nearest Neighbors", test_classifier(flare, knn, .25))
