### The difference between this file and HelloLLAVA.py is that this one is newer
### I got tired of editing HellovLlava.py and wanted to start fresh

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import os
import json
from tqdm import tqdm
from itertools import product
from sklearn.linear_model import LogisticRegression

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directories for saving results
os.makedirs("./LLaVA_results/line_big", exist_ok=True)
os.makedirs("./LLaVA_results/line_big/before_projection", exist_ok=True)
os.makedirs("./LLaVA_results/line_big/after_projection", exist_ok=True)

def record_misclassifications(predictions, true_labels, identifiers, split, trained_dataset_num, test_dataset_num, classifier_name, projected):
    misclassified = []
    for pred, true, id in zip(predictions, true_labels, identifiers):
        if pred != true:
            misclassified.append({
                'identifier': id,
                'true_label': int(true),
                'predicted_label': int(pred)
            })
    
    # Save misclassifications to a JSON file
    result_type = "after_projection" if projected else "before_projection"
    filename = f"./LLaVA_results/line_big/{result_type}/misclassifications_trained_{trained_dataset_num}_tested_{test_dataset_num}_{classifier_name}_{split}.json"
    with open(filename, 'w') as f:
        json.dump(misclassified, f, indent=2)
    
    print(f"Misclassifications for trained on Dataset {trained_dataset_num} tested on Dataset {test_dataset_num} {split} set saved to {filename}")
    return len(misclassified)

# Function to load and preprocess features
def load_and_preprocess_features(csv_file, feature_dir, projected=False):
    data = pd.read_csv(csv_file)
    identifier_column = data.columns[0]  # Assume the first column is the identifier
    features = []
    labels = []

    for _, row in tqdm(data.iterrows(), total=len(data), desc=f"Loading features from {os.path.basename(csv_file)}"):
        feature_path = os.path.join(
            feature_dir, 
            f"{row[identifier_column]}_{('projected' if projected else 'vision')}_features.pt"
        )
        if os.path.exists(feature_path):
            try:
                feature = torch.load(feature_path, map_location=device)
                # Apply max pooling along the first and last dimensions
                # feature = torch.max(torch.max(feature, dim=0)[0], dim=-0)[0]
                #feature = torch.mean(torch.mean(feature, dim=0), dim=0)
                while feature.dim() > 1:
                    feature = torch.mean(feature, dim=0)
                features.append(feature.cpu().numpy())
                labels.append(row["label"])
            except Exception as e:
                print(f"Error loading file: {feature_path}")
                print(f"Error message: {str(e)}")
        else:
            print(f"Feature not found: {feature_path}")

    features = np.array(features)
    labels = np.array(labels)
    print(f"Features shape from {os.path.basename(csv_file)}: {features.shape}")
    print(f"Labels shape from {os.path.basename(csv_file)}: {labels.shape}")

    return features, labels

# Function to calculate accuracy per distance
def calculate_accuracy_per_distance(model, features, labels, distances):
    # idc
    return {}
    predictions = model.predict(features)
    unique_distances = sorted(set(distances))
    accuracies = {}
    for dist in unique_distances:
        mask = distances == dist
        acc = accuracy_score(labels[mask], predictions[mask])
        accuracies[float(dist)] = float(acc)
    return accuracies

def process_llava_features(projected=False, num_datasets=5):
    feature_dir = "/aiau001_scratch/logan/line_features"

    results = {}
    overall_results = {}
    misclassification_counts = {}

    # Load metadata once
    metadata = pd.read_json("./line_dataset/metadata.json", orient='index')

    # Define hyperparameter grid for MLPClassifier
    mlp_param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 100, 50)],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [500, 1000]
    }
#
    # Generate all combinations of hyperparameters
    mlp_param_combinations = list(product(
        mlp_param_grid['hidden_layer_sizes'],
        mlp_param_grid['learning_rate_init'],
        mlp_param_grid['alpha'],
        mlp_param_grid['max_iter']
    ))
#
    print(f"\nProcessing LLaVA features {'after' if projected else 'before'} projection...")
    #for trained_i in {1, 3, 5, 7, 9}:
    for trained_i in {99999999}:
    #for trained_i in range(1, num_datasets + 1):
        print(f"\nTraining on Dataset {trained_i}...")
        dataset_results = {}
        dataset_overall_results = {}
        dataset_misclassification_counts = {}
#
        ## Load and preprocess training and validation data for trained_i
        #train_csv = f"./line_dataset/splits/train_{trained_i}-{trained_i+2}.csv"
        train_csv = f"./line_dataset/splits/train_all.csv"
        #val_csv = f"./datasets/dataset_{trained_i}_val.csv"
        val_csv = f"./line_dataset/splits/val_all.csv"
        #val_csv = f"./line_dataset/splits/val_{trained_i}-{trained_i+2}.csv"
        train_features, train_labels = load_and_preprocess_features(train_csv, feature_dir, projected)
        valid_features, valid_labels = load_and_preprocess_features(val_csv, feature_dir, projected)

        ## Standardize the features based on training data
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        valid_features_scaled = scaler.transform(valid_features)
#
        ## Hyperparameter tuning
        best_val_accuracy = 0
        #best_params = None
#
        #print(f"Starting hyperparameter tuning for training on Dataset {trained_i}...")
        #for params in tqdm(mlp_param_combinations, desc="Tuning hyperparameters"):
            #hidden_layer_sizes, learning_rate_init, alpha, max_iter = params
            #classifier = MLPClassifier(
                #hidden_layer_sizes=hidden_layer_sizes,
                #learning_rate_init=learning_rate_init,
                #alpha=alpha,
                #random_state=42,
                #max_iter=max_iter
            #)
            #try:
                #classifier.fit(train_features_scaled, train_labels)
                #val_predictions = classifier.predict(valid_features_scaled)
                #val_accuracy = accuracy_score(valid_labels, val_predictions)
#
                #if val_accuracy > best_val_accuracy:
                    #best_val_accuracy = val_accuracy
                    #best_params = {
                        #'hidden_layer_sizes': hidden_layer_sizes,
                        #'learning_rate_init': learning_rate_init,
                        #'alpha': alpha,
                        #'max_iter': max_iter
                    #}
            #except Exception as e:
                #print(f"Training failed for parameters {params}: {e}")
                #continue
#        best_params = {
#            'hidden_layer_sizes': (),
#            'learning_rate_init': 0.001,
#            'alpha': 0.0001,
#            'max_iter': 500
#        }
##
#        print(f"Best validation accuracy for training on Dataset {trained_i}: {best_val_accuracy:.4f} with params: {best_params}")
#
#        # Retrain the classifier with the best hyperparameters on the training set
#        best_classifier = MLPClassifier(
#            hidden_layer_sizes=(),
#            learning_rate_init=best_params['learning_rate_init'],
#            activation='identity',
#            alpha=best_params['alpha'],
#            random_state=42,
#            max_iter=best_params['max_iter']
#        )

        best_params = {
            'C': 1.0,  # Regularization parameter
            'max_iter': 1000,  # Maximum iterations
            'solver': 'lbfgs'  # Solver type
        }


        # Initialize and train LogisticRegression
        best_classifier = LogisticRegression(
            C=best_params['C'],
            max_iter=best_params['max_iter'],
            solver=best_params['solver'],
            random_state=42,
            n_jobs=-1  # Use all available cores
        )

        best_classifier.fit(train_features_scaled, train_labels)

        best_classifier.fit(train_features_scaled, train_labels)

        # Initialize results for this trained model
        dataset_results["MLPClassifier"] = {}
        dataset_overall_results["MLPClassifier"] = {}
        dataset_misclassification_counts["MLPClassifier"] = {}

        # Iterate over all test datasets
        #for tested_j in {1, 3, 5, 7, 9}:
        for tested_j in {999999999}:
        #for tested_j in range(1, num_datasets + 1):
            print(f"\nTesting model trained on Dataset {trained_i} on Test Dataset {tested_j}...")
            #test_csv = f"./datasets/dataset_{tested_j}_test.csv"
            #test_csv = f"./line_dataset/splits/test_{tested_j}-{tested_j+2}.csv"
            test_csv = f"./line_dataset/splits/test_all.csv"
            test_features, test_labels = load_and_preprocess_features(test_csv, feature_dir, projected)

            # Standardize the test features using the scaler fitted on training data
            test_features_scaled = scaler.transform(test_features)

            # Predict
            test_predictions = best_classifier.predict(test_features_scaled)

            # Calculate overall accuracy
            overall_acc = accuracy_score(test_labels, test_predictions)
            print(f"  Test Dataset {tested_j} Overall Accuracy: {overall_acc:.4f}")

            # Calculate accuracy per distance
            # Assuming that 'distance' is relevant for test datasets; adjust if necessary
            csv_data = pd.read_csv(test_csv)
            identifier_column = csv_data.columns[0]
            #distances = metadata.loc[csv_data[identifier_column], 'distance'].values
            #accuracies_per_distance = calculate_accuracy_per_distance(best_classifier, test_features_scaled, test_labels, distances)
            #print(f"  Test Dataset {tested_j} Accuracies per Distance: {accuracies_per_distance}")

            # Record accuracies
            #dataset_results["MLPClassifier"][f"Test_Dataset_{tested_j}"] = accuracies_per_distance
            dataset_overall_results["MLPClassifier"][f"Test_Dataset_{tested_j}"] = float(overall_acc)

            # Record misclassifications
            misclassification_count = record_misclassifications(
                test_predictions, test_labels, csv_data[identifier_column].values, "test", 
                trained_i, tested_j, "MLPClassifier", projected
            )
            dataset_misclassification_counts["MLPClassifier"][f"Test_Dataset_{tested_j}"] = misclassification_count

        # Save results for this trained model
        results[f"Trained_on_Dataset_{trained_i}"] = dataset_results
        overall_results[f"Trained_on_Dataset_{trained_i}"] = dataset_overall_results
        misclassification_counts[f"Trained_on_Dataset_{trained_i}"] = dataset_misclassification_counts

    # Save detailed results
    result_type = "after_projection" if projected else "before_projection"
    filename = f"./LLaVA_results/line_big/{result_type}/classification_results.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed classification results saved to {filename}")

    # Save overall results
    overall_filename = f"./LLaVA_results/line_big/{result_type}/overall_classification_results.json"
    with open(overall_filename, "w") as f:
        json.dump(overall_results, f, indent=2)
    print(f"Overall classification results saved to {overall_filename}")

    # Save misclassification counts
#    misclassification_filename = f"./LLaVA_results/{result_type}/misclassification_counts.json"
#    with open(misclassification_filename, "w") as f:
#        json.dump(misclassification_counts, f, indent=2)
#    print(f"Misclassification counts saved to {misclassification_filename}")
#
# Process features before and after projection
print("Processing LLaVA features before projection...")
process_llava_features(projected=False, num_datasets=5)

print("\nProcessing LLaVA features after projection...")
process_llava_features(projected=True, num_datasets=5)

print("\nAll processing completed.")

