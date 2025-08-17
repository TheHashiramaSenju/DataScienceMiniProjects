# Data Science Mini Projects 🧠📊

[![GitHub stars](https://img.shields.io/github/stars/TheHashiramaSenju/DataScienceMiniProjects?style=flat-square)](https://github.com/TheHashiramaSenju/DataScienceMiniProjects/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/TheHashiramaSenju/DataScienceMiniProjects?style=flat-square)](https://github.com/TheHashiramaSenju/DataScienceMiniProjects/network)
[![GitHub issues](https://img.shields.io/github/issues/TheHashiramaSenju/DataScienceMiniProjects?style=flat-square)](https://github.com/TheHashiramaSenju/DataScienceMiniProjects/issues)
[![Contributors](https://img.shields.io/github/contributors/TheHashiramaSenju/DataScienceMiniProjects?style=flat-square)](https://github.com/TheHashiramaSenju/DataScienceMiniProjects/graphs/contributors)
[![License: MIT](https://img.shields.io/github/license/TheHashiramaSenju/DataScienceMiniProjects?style=flat-square)](LICENSE)
[![Repo size](https://img.shields.io/github/repo-size/TheHashiramaSenju/DataScienceMiniProjects?style=flat-square)]()

---

## Table of Contents 

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Automated Setup Script](#automated-setup-script)
- [Usage](#usage)
- [Git Initialization & Push](#git-initialization--push)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

This repository hosts a collection of **Supervised** and **Unsupervised Learning** mini-projects, each organized into:

- **Concepts** (`concepts.md`) – Core explanations  
- **Intricacies** (`intricacies.md`) – Deep-dive technical details  
- **Implementation** (`<Concept>.py`) – Python code  

Use this as an educational reference or a starting point for your own data-science workflows.

---

## Directory Structure

Below is the fully expanded tree under `Datascience/`. Every method has its own subfolder with three files:  
`concepts.md`, `intricacies.md`, and `<MethodName>.py`.

```bash
Datascience/
├── SupervisedLearning/
│   ├── Regression/
│   │   ├── LinearRegression/
│   │   │   ├── concepts.md
│   │   │   ├── intricacies.md
│   │   │   └── LinearRegression.py
│   │   ├── LogisticRegression/
│   │   │   ├── concepts.md
│   │   │   ├── intricacies.md
│   │   │   └── LogisticRegression.py
│   │   ├── RidgeRegression/
│   │   │   ├── concepts.md
│   │   │   ├── intricacies.md
│   │   │   └── RidgeRegression.py
│   │   ├── LassoRegression/
│   │   │   ├── concepts.md
│   │   │   ├── intricacies.md
│   │   │   └── LassoRegression.py
│   │   └── PolynomialRegression/
│   │       ├── concepts.md
│   │       ├── intricacies.md
│   │       └── PolynomialRegression.py
│   ├── Classification/
│   │   ├── DecisionTrees/
│   │   │   ├── concepts.md
│   │   │   ├── intricacies.md
│   │   │   └── DecisionTrees.py
│   │   ├── RandomForest/
│   │   │   ├── concepts.md
│   │   │   ├── intricacies.md
│   │   │   └── RandomForest.py
│   │   ├── SupportVectorMachines/
│   │   │   ├── concepts.md
│   │   │   ├── intricacies.md
│   │   │   └── SupportVectorMachines.py
│   │   └── NaiveBayes/
│   │       ├── concepts.md
│   │       ├── intricacies.md
│   │       └── NaiveBayes.py
│   └── NeuralNetworks/
│       ├── Perceptron/
│       │   ├── concepts.md
│       │   ├── intricacies.md
│       │   └── Perceptron.py
│       ├── ConvolutionalNeuralNetworks/
│       │   ├── concepts.md
│       │   ├── intricacies.md
│       │   └── ConvolutionalNeuralNetworks.py
│       └── RecurrentNeuralNetworks/
│           ├── concepts.md
│           ├── intricacies.md
│           └── RecurrentNeuralNetworks.py
├── UnsupervisedLearning/
│   ├── Clustering/
│   │   ├── KMeans/
│   │   │   ├── concepts.md
│   │   │   ├── intricacies.md
│   │   │   └── KMeans.py
│   │   ├── HierarchicalClustering/
│   │   │   ├── concepts.md
│   │   │   ├── intricacies.md
│   │   │   └── HierarchicalClustering.py
│   │   └── DBSCAN/
│   │       ├── concepts.md
│   │       ├── intricacies.md
│   │       └── DBSCAN.py
│   ├── DimensionalityReduction/
│   │   ├── PrincipalComponentAnalysis/
│   │   │   ├── concepts.md
│   │   │   ├── intricacies.md
│   │   │   └── PrincipalComponentAnalysis.py
│   │   └── SingularValueDecomposition/
│   │       ├── concepts.md
│   │       ├── intricacies.md
│   │       └── SingularValueDecomposition.py
│   ├── AnomalyDetection/
│   │   ├── IsolationForest/
│   │   │   ├── concepts.md
│   │   │   ├── intricacies.md
│   │   │   └── IsolationForest.py
│   │   └── LocalOutlierFactor/
│   │       ├── concepts.md
│   │       ├── intricacies.md
│   │       └── LocalOutlierFactor.py
│   └── AssociationRules/
│       ├── Apriori/
│       │   ├── concepts.md
│       │   ├── intricacies.md
│       │   └── Apriori.py
│       └── Eclat/
│           ├── concepts.md
│           ├── intricacies.md
│           └── Eclat.py
└── struct.sh
