# Title     : TODO
# Objective : TODO
# Created by: bryanmuller
# Created on: 2019-03-01
#The first time through, you'll need to install the LIBSVM package:
# install.packages('e1071', dependencies = TRUE);

# Include the LIBSVM package
library(e1071)
require(stats)

###############
# FUNCTIONS
###############
partitionData <- function(data, testSize, classCol) {
    data[, -classCol] <- scale(data[, -classCol])
    split = {}
    allRows <- 1:nrow(data)
    testRows = complete.cases(sample(allRows, trunc(length(allRows) * testSize)))
    split$Test = data[testRows,]
    split$Train = data[-testRows,]
    return(split)
}
# Load our old friend, the Iris data set
# Note that it is included in the default datasets library
# library(datasets)
# data(iris)

# For your assignment, you'll need to read from a CSV file.
# Conveniently, there is a read.csv() function that can be used like so:
# myDataSet <- read.csv(fileName, head=TRUE, sep=",")
print("testing vowels")
vowels <- read.csv('./vowel.csv', head=TRUE, sep=",", stringsAsFactors=TRUE)
vowels$Sex <- as.numeric(vowels$Sex)
vowels$Speaker <- as.numeric(vowels$Speaker)
partitionedVowels <- partitionData(vowels, 0.3, 13)
bestVowelAgreement <- 0
bestVowelGamma <- 0
bestVowelC <- 0
bestVowelAccuracy <- 0
allRows <- 1:nrow(vowels)

testVowelModel <- function (vowels, k, g, cost, errChange) {
    model <- svm(Class~., data = vowels$Train, kernel = k, gamma = g, cost = cost)
    prediction <- predict(model, vowels$Test)
    confusionMatrix <- table(pred = prediction, true = vowels$Test$Class)
    agreement <- prediction == vowels$Test$Class
    accuracy <- prop.table(table(agreement))
    truePred <- ifelse(length(table(agreement)) == 1, table(agreement)[1], table(agreement)[2])
    err <- (truePred - bestVowelAgreement) / length(allRows)
    if (truePred > bestVowelAgreement && (err > errChange)) {
        bestVowelAgreement <<- truePred
        bestVowelC <<- cost
        bestVowelGamma <<- g
        bestVowelAccuracy <<- accuracy
    }
}

for (cost in seq(1, 3)) {
    for (gammat in seq(.01, 1, by=.1)) {
        testVowelModel(partitionedVowels, k = "radial", g = gammat, cost = cost, errChange=.1)
    }
}
print("Best Values for Gamma and Cost with accompanying accuracy")
print("Agreement")
print(bestVowelAgreement)
print("Accuracy")
print(bestVowelAccuracy)
print("Cost")
print(bestVowelC)
print("Gamma")
print(bestVowelGamma)

