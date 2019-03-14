# Title     : TODO
# Objective : TODO
# Created by: bryanmuller
# Created on: 2019-03-13

###########################
# KMeans
########################

# Step 1
##########
data <- scale(state.x77)
# Step 2
#########
myClusters = kmeans(data, 3)

# Summary of the clusters
summary(myClusters)
# Length Class  Mode
# cluster      50     -none- numeric
# centers      24     -none- numeric
# totss         1     -none- numeric
# withinss      3     -none- numeric
# tot.withinss  1     -none- numeric
# betweenss     1     -none- numeric
# size          3     -none- numeric
# iter          1     -none- numeric
# ifault        1     -none- numeric

# Centers (mean values) of the clusters
print(myClusters$centers)
# Population     Income   Illiteracy   Life Exp     Murder    HS Grad    Frost       Area
# 1 -0.2269956 -1.3014617  1.391527063 -1.1773136  1.0919809 -1.4157826 -0.7206500 -0.2340290
# 2 -0.4873370  0.1329601 -0.641201154  0.7422562 -0.8552439  0.5515044  0.4528591 -0.1729366
# 3  0.9462026  0.7416690  0.005468667 -0.3242467  0.5676042  0.1558335 -0.1960979  0.4483198

# Cluster assignments
print(myClusters$cluster)
# Alabama         Alaska        Arizona       Arkansas     California
# 3              1              1              3              1
# Colorado    Connecticut       Delaware        Florida        Georgia
# 2              2              2              1              3
# Hawaii          Idaho       Illinois        Indiana           Iowa
# 2              2              1              2              2
# Kansas       Kentucky      Louisiana          Maine       Maryland
# 2              3              3              2              1
# Massachusetts       Michigan      Minnesota    Mississippi       Missouri
# 2              1              2              3              1
# Montana       Nebraska         Nevada  New Hampshire     New Jersey
# 2              2              1              2              1
# New Mexico       New York North Carolina   North Dakota           Ohio
# 3              1              3              2              1
# Oklahoma         Oregon   Pennsylvania   Rhode Island South Carolina
# 2              2              1              2              3
# South Dakota      Tennessee          Texas           Utah        Vermont
# 2              3              1              2              2
# Virginia     Washington  West Virginia      Wisconsin        Wyoming
# 1              2              3              2              2

# Plotting a visual representation of k-means clusters
library(cluster)
png("k-3-clusters.png")
clusplot(data, myClusters$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)

# Step 3
###############
errors <- c()
for (k in 1:25) {
    errors[k] <- kmeans(data, k)$tot.withinss
}
png("kmeans-sum-squares-error.png")
plot(errors, xlab = "k", ylab = "within-cluster sum of squares error")

# Step 4
################

# From elbow method
k <- 10

cluster = kmeans(data, k)

# Step 5
###############

print(cluster$cluster)
# Alabama         Alaska        Arizona       Arkansas     California
# 4              2              3              4             10
# Colorado    Connecticut       Delaware        Florida        Georgia
# 7              1              8              8              4
# Hawaii          Idaho       Illinois        Indiana           Iowa
# 5              7              8              8              7
# Kansas       Kentucky      Louisiana          Maine       Maryland
# 7              4              4              9              8
# Massachusetts       Michigan      Minnesota    Mississippi       Missouri
# 1              8              7              4              8
# Montana       Nebraska         Nevada  New Hampshire     New Jersey
# 6              7              6              9              8
# New Mexico       New York North Carolina   North Dakota           Ohio
# 3             10              4              1              8
# Oklahoma         Oregon   Pennsylvania   Rhode Island South Carolina
# 3              5              8              1              4
# South Dakota      Tennessee          Texas           Utah        Vermont
# 9              4             10              7              9
# Virginia     Washington  West Virginia      Wisconsin        Wyoming
# 8              5              4              7              6


# Step 6
###############
png("kmeans-10.png")
clusplot(data, cluster$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)

# Step 7
##############
print(cluster$centers)

   # Population     Income Illiteracy   Life Exp     Murder     HS Grad  Frost         Area
# 1  -0.3499380  0.5656501 -0.7710820  1.2544011 -1.1080742  0.55150442  0.859258777 -0.058630181
# 2  -0.4514893  0.5182516  0.0902330  0.8353735 -0.4748696  0.96161967 -1.571925102 -0.001018197
# 3  -0.7660428 -0.5843829 -0.9117048  0.4809958 -0.9150653  0.65342525  0.994267293 -0.099820942
# 4  -0.1667872 -1.3624751  1.8866900 -1.7868083  1.5933731 -1.55107136 -1.213139113 -0.287006387
# 5  -0.8429672  0.6862826 -1.0171720 -0.9077815  0.4935610  1.35471127  1.462846466  0.384520782
# 6   0.7891560  0.5328170 -0.3117140 -0.2462765  0.3093560 -0.19041729 -0.001154271 -0.342772830
# 7   2.8948232  0.4869237  0.6507713  0.1301655  1.0172810  0.13932569 -1.131057600  0.992720037
# 8  -0.2771693 -1.2506172  0.9788913 -0.6694013  0.6741541 -1.30304191 -0.310242469 -0.189881160
# 9  -0.8693980  3.0582456  0.5413980 -1.1685098  1.0624293  1.68280347  0.914567609  5.809349671
# 10 -0.3889962  0.1472000 -0.1148420  0.3157792 -0.7593038 -0.04122819 -0.013658877 -0.595660829






