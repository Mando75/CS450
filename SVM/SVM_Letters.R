# Title     : TODO
# Objective : TODO
# Created by: bryanmuller
# Created on: 2019-03-02
library(e1071)
require(stats)

partitionData <- function(data, testSize, classCol) {
    data[, -classCol] <- scale(data[, -classCol])
    split = {}
    allRows <- 1:nrow(data)
    testRows = complete.cases(sample(allRows, trunc(length(allRows) * testSize)))
    split$Test = data[testRows,]
    split$Train = data[-testRows,]
    return(split)
}

print("testing letter")
letters <- read.csv("./letters.csv")
partitionedLetters <- partitionData(letters, 0.3, 1)
bestLetterAgreement <- 0
bestLetterC <- 0
bestLetterGamma <- 0
bestLetterC <- 0
bestLetterAccuracy <- 0
allRows <- 1:nrow(letters)

testLetterModel <- function (letters, k, g, cost, errChange) {
    model <- svm(letter ~ ., data = letters$Train, kernel = k, gamma = g, cost = cost)
    prediction <- predict(model, letters$Test)
    confusionMatrix <- table(pred = prediction, true = letters$Test$letter)
    agreement <- prediction == letters$Test$letter
    accuracy <- prop.table(table(agreement))
    truePred <- ifelse(length(table(agreement)) == 1, table(agreement)[1], table(agreement)[2])
    err <- (truePred - bestLetterAgreement) / length(allRows)
    if (truePred > bestLetterAgreement && (err > errChange)) {
        bestLetterAgreement <<- truePred
        bestLetterC <<- cost
        bestLetterGamma <<- g
        bestLetterAccuracy <<- accuracy
    }
}

# This loop takes awhile to run. I noticed that the model takes awhile
# to train. The slowdown is in the svm() function in testLetterModel
for (cost in seq(1, 3)) {
    for (gammat in seq(.01, 1, by=.1)) {
        testLetterModel(partitionedLetters, k = "radial", g = gammat, cost = cost, errChange=.1)
    }
}
print("Best Values for Gamma and Cost with accompanying accuracy")
print("Agreement")
print(bestLetterAgreement)
print("Accuracy")
print(bestLetterAccuracy)
print("Cost")
print(bestLetterC)
print("Gamma")
print(bestLetterGamma)
