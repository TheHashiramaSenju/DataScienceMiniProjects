#!/bin/bash

# Define the base directory
BASE_DIR="/home/user/Documents/MiscFiles/Datascience"

# Ensure the directory exists
mkdir -p "$BASE_DIR"

# Define subfolders and their specialized categories
declare -A supervised_learning
supervised_learning["Regression"]="LinearRegression LogisticRegression RidgeRegression LassoRegression PolynomialRegression"
supervised_learning["Classification"]="DecisionTrees RandomForest SupportVectorMachines NaiveBayes"
supervised_learning["NeuralNetworks"]="Perceptron ConvolutionalNeuralNetworks RecurrentNeuralNetworks"

declare -A unsupervised_learning
unsupervised_learning["Clustering"]="KMeans HierarchicalClustering DBSCAN"
unsupervised_learning["DimensionalityReduction"]="PrincipalComponentAnalysis SingularValueDecomposition"
unsupervised_learning["AnomalyDetection"]="IsolationForest LocalOutlierFactor"
unsupervised_learning["AssociationRules"]="Apriori Eclat"

# Function to create directories & files properly
create_structure() {
    local parent_folder="$BASE_DIR/$1"
    shift
    declare -n concepts=$1

    mkdir -p "$parent_folder"

    for concept in "${!concepts[@]}"; do
        mkdir -p "$parent_folder/$concept"
        for subcategory in ${concepts[$concept]}; do
            mkdir -p "$parent_folder/$concept/$subcategory"
            touch "$parent_folder/$concept/$subcategory/concepts.md"
            touch "$parent_folder/$concept/$subcategory/intricacies.md"
            touch "$parent_folder/$concept/$subcategory/${subcategory}.py"
        done
    done
}

# Create the directory structure correctly
create_structure "SupervisedLearning" supervised_learning
create_structure "UnsupervisedLearning" unsupervised_learning

echo "Full directory structure successfully created inside $BASE_DIR 🚀"

