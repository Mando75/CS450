from load_semeion import load_semeion

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier


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

########################
# Datasets
#######################
semeion = load_semeion()

print("|   Dataset   |   Classifier   |   Accuracy   |")
print("--------------+----------------+--------------+")
printResults("Semeion", "Neural Network", test_classifier(semeion, nn, .25))
