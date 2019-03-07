# Title     : TODO
# Objective : TODO
# Created by: bryanmuller
# Created on: 2019-03-05
# install.packages('arules');
library(arules);
data(Groceries);

# Highest support I could find
print("Highest support")
highestSupport <- apriori(Groceries, parameter = list(supp=0.1, conf=0.0))
highestSupport <- sort(highestSupport, decreasing = TRUE, by = "support")
inspect(highestSupport[1:5])
print('####################################################################')

# Highest confidence I could find
print("Highest Confidence")
hConfidence <- apriori(Groceries, parameter = list(supp=.001, conf=1))
hConfidence <- sort(hConfidence, decreasing = TRUE, by = "confidence")
inspect(hConfidence[1:5])
print('####################################################################')

# Highest lift I could find
print("Highest Lift")
hLift <- apriori(Groceries, parameter = list(supp = 0.001, conf = 0.1))
hLift <- sort(hLift, decreasing = TRUE, by = "lift")
inspect(hLift[1:5])
print('###################################################################')

# Coolest rules I could find
print("Most interesting")
rules <- apriori(Groceries, parameter = list(supp = 0.0005, conf = 1))
rules <- sort(rules, decreasing = TRUE, by = "lift")
interesting <- c(13, 25, 170, 366,385)
inspect(rules[interesting])

# This should work, but the tcltk package isn't working on my Mac... Haven't been able to figure out why
# library(arulesViz)
# plot(rules[1:10], method = "graph", engine = "interactive", shading = NA)


