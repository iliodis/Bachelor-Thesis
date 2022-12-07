# Bachelor-Thesis

## Introduction
My Bachelor Thesis is focused on using **Artificial Neural Networks** for the Mass-Radius relation of Neutron Stars in **General Relativity** and **Alternative Theories of Gravity**.

I conduct my Thesis with **Professor Nikolaos Stergioulas** in Aristotle University of Thessaloniki, in his **Gravitational Waves Group**. In this repository I record my progress in this project. Here, I upload the data used and the code written. Additionally, I upload the presentations I give to the Gravitational Waves Group for my progress.

## Organisation
There are three important folders called "Gravitational Waves Group Presentations", "Code" and "Data".

The organisation of the code follows a timelike order to comply with the presentations. The folder "Code" consist of several folders called "Phase n" (where n is a Natural number) corresponding to "Progress Presentation n" in the "Gravitational Waves Group Presentations" folder. Each progress presentation consists of a brief theoretical introduction and a review of the main results, while each group of notebooks or other code, related to Presentation n, is uploaded in the folder "Phase n". Finally, in the "Data" folder I upload the data used in the notebooks.

## Phases
During the first phase I had been focused on finding the best tuning for Neutron Star data in GR and for two Equations of State (EOS) (piecewise polytropic SLy and a tabulated). The best model had achieved a final loss below $10^{-4}$!

After the first presentation I started using data from the Einstein-Gauss-Bonnet Gravity and for the SLy EOS. These data increase the input parameters by one, so I firstly focused on Mass-Radius curves with a constant alpha value. By the end of the second phase, and regarding the one dimensional input, I achieved to stabilize the training of the best models found previously and to decrease the Mean Square Error Loss by several orders of magnitude by implementing tricks and techniques for increased precision. With only three hidden layers the best model achieved a Mean Absolute Percentage Error of the order of $10^{-3}$ %!

![BGFS_triple](https://user-images.githubusercontent.com/80003772/206258460-c71b1137-2d50-40a7-8e3b-c92ea93ff3e2.png)

Some first trials when fitting the whole dataset showed some very interesting results as well!

![fitalldata](https://user-images.githubusercontent.com/80003772/206258718-97e9cf80-79d7-42ee-90dc-ee21cb5dd49f.png)
