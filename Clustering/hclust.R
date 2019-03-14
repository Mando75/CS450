# Title     : TODO
# Objective : TODO
# Created by: bryanmuller
# Created on: 2019-03-13
###############################
# AH Clustering
###############################
# Load the dataset
library(datasets)
library(cluster)
data = state.x77

# Use hierarchical clustering to cluster the data on all attributes and produce a dendrogram
png(filename="hclust-all-attri-no-scale.png")
plot(hclust(dist(as.matrix(data))))

# Repeat the previous item with a normalized dataset and note any differences
png(filename="hclust-all-attri-with-scale.png")
plot(hclust(dist(as.matrix(scale(data)))))

# Remove "Area" from the attributes and re-cluster (and note any differences)
png(filename="hclust-no-area-with-scale.png")
plot(hclust(dist(as.matrix(scale(data[, c("Population", "Income", "Illiteracy", "Life Exp", "Murder", "HS Grad", "Frost")])))))

# Cluster only on the Frost attribute and observe the results
png(filename="hclust-only-frost-with-scale.png")
plot(hclust(dist(as.matrix(scale(data[,"Frost"])))))


