# ImpKmeans
Python Implementation of ImpKmeans, which is an advanced version of Lloyd's k-means algorithm

K-means is the most known clustering algorithm because of its usage simplicity, speed, and efficiency. However, the resultant clusters are influenced by the randomly selected initial centroids. Therefore, many techniques have been implemented to solve the mentioned issue. In this paper, a new version of the k-means clustering algorithm named as ImpKmeans shortly (An Improved Version of K-Means Algorithm by Determining Optimum Initial Centroids Based on Multivariate Kernel Density Estimation and Kd-tree) that uses kernel density estimation to find the optimum initial centroids is proposed. Kernel density estimation is used because it is a nonparametric distribution estimation method that can find density regions.

To understand the efficiency of ImpKmeans, we compared it with classical k-means, k-means++, PAM, and Fuzzy k-means algorithms. Some of the obtained results of the experimental study are shared below. 

![2d-20c](https://github.com/senolali/ImpKmeans/assets/72247990/f373465b-93ae-4819-9350-2196d8248488)

<p align="center">
<img src="https://github.com/senolali/ImpKmeans/Datasets/2d-20c_KDE.png"  width="200" height="400" />
</p>
![Aggregation](https://github.com/senolali/ImpKmeans/assets/72247990/d45ea3f9-06b4-4c99-912a-753d83592167)

![Aggregation_KDE](https://github.com/senolali/ImpKmeans/assets/72247990/b55c0edf-8818-4d87-b3a0-3bee369df75a)

If you are interested in the algorithm used in your study, please refer to the article shared below:

Şenol, A., (2023). "ImpKmeans: An Improved Version of K-Means Algorithm by Determining Optimum Initial Centroids Based on Multivariate Kernel Density Estimation and Kd-Tree", Acta Polytechnica Hungarica.

