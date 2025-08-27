
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import os
import json
from tqdm import tqdm
from itertools import product
from sklearn.linear_model import LogisticRegression


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

save_string = "LR_BIG"
# Create directories for saving results
os.makedirs("./Phi_results", exist_ok=True)
os.makedirs(f"./Phi_results/line_{save_string}/before_projection", exist_ok=True)
os.makedirs(f"./Phi_results/line_{save_string}/after_projection", exist_ok=True)

# Function to load and preprocess features
def load_and_preprocess_features(csv_file, feature_dir, projected=False):
    data = pd.read_csv(csv_file)
    identifier_column = data.columns[0]  # Assume the first column is the identifier
    features = []
    labels = []

    for _, row in tqdm(data.iterrows(), total=len(data), desc=f"Loading features from {os.path.basename(csv_file)}"):
        #   f"{row[identifier_column]}_{('projected' if projected else 'vision')}_features.pt"
        feature_path = os.path.join(
            feature_dir, 
            f"{row[identifier_column]}_phi3_{('after' if projected else 'before')}_projection.pt"
        )
        if os.path.exists(feature_path):
            try:
                feature = torch.load(feature_path, map_location=device)
                # Apply max pooling along the first and last dimensions
                # feature = torch.max(torch.max(feature, dim=0)[0], dim=-0)[0]

                
                #if projected:
                if False:
                    # For SigLIP: [batch_size, num_patches, hidden_size] -> [hidden_size]
                    feature = torch.mean(torch.mean(feature, dim=0), dim=0)
                else:
                    while feature.dim() > 1:
                        feature = torch.mean(feature, dim=0)
                    # Ensure feature is 1D
                    if feature.dim() != 1:
                        raise ValueError(f"Unexpected feature dimensions: {feature.shape}")
                
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
    predictions = model.predict(features)
    unique_distances = sorted(set(distances))
    accuracies = {}
    for dist in unique_distances:
        mask = distances == dist
        acc = accuracy_score(labels[mask], predictions[mask])
        accuracies[float(dist)] = float(acc)
    return accuracies

def process_phi_features(projected=False, num_datasets=5):
    feature_dir = "/aiau001_scratch/logan/phi/line_features"

    results = {}
    overall_results = {}
    misclassification_counts = {}

    print(f"\nProcessing Phi features {'after' if projected else 'before'} projection...")
    
    # for trained_i in range(1, num_datasets + 1):
    #for i in range(1, 11, 2):
    for i in {8888888}:
        print(f"\nTraining on Dataset {i}-{i+2}...")
        dataset_results = {}
        dataset_overall_results = {}
        dataset_misclassification_counts = {}

        # Load and preprocess training and validation data for trained_i
        #train_csv = f"./line_dataset/splits/train_{i}-{i+2}.csv"
        train_csv = f"./line_dataset/splits/train_all.csv"
        #val_csv = f"./line_dataset/splits/val_{i}-{i+2}.csv"
        val_csv = f"./line_dataset/splits/val_all.csv"
        train_features, train_labels = load_and_preprocess_features(train_csv, feature_dir, projected)
        valid_features, valid_labels = load_and_preprocess_features(val_csv, feature_dir, projected)

        # Standardize the features based on training data
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        valid_features_scaled = scaler.transform(valid_features)

        #best_params = {
            #'hidden_layer_sizes': (50, 50),
            #'learning_rate_init': 0.001,
            #'alpha': 0.0001,
            #'max_iter': 1500
        #}
#
        #print(f"Dataset {i}-{i+2} with params: {best_params}")
#
        ## Retrain the classifier with the best hyperparameters on the training set
        #best_classifier = MLPClassifier(
            #hidden_layer_sizes=best_params['hidden_layer_sizes'],
            #activation='identity',
            #learning_rate_init=best_params['learning_rate_init'],
            #alpha=best_params['alpha'],
            #random_state=42,
            #max_iter=best_params['max_iter']
        #)

        best_params = {
            'C': 1.0,  # Regularization parameter
            'max_iter': 1000,  # Maximum iterations
            'solver': 'lbfgs'  # Solver type
        }

        print(f"Dataset {i}-{i+2} with params: {best_params}")

        # Initialize and train LogisticRegression
        best_classifier = LogisticRegression(
            C=best_params['C'],
            max_iter=best_params['max_iter'],
            solver=best_params['solver'],
            random_state=42,
            n_jobs=-1  # Use all available cores
        )

        best_classifier.fit(train_features_scaled, train_labels)

        # Initialize results for this trained model
        dataset_results["MLPClassifier"] = {}
        dataset_overall_results["MLPClassifier"] = {}
        dataset_misclassification_counts["MLPClassifier"] = {}

        # Iterate over all test datasets
        # for tested_j in range(1, num_datasets + 1):
        #for j in range(1, 11, 2):
        for j in {8888888}:
            print(f"\nTesting model trained on Dataset {i}-{i+2} on Test Dataset {j}-{j+2}...")
            #test_csv = f"./line_dataset/splits/test_{j}-{j+2}.csv"
            test_csv = f"./line_dataset/splits/test_all.csv"
            test_features, test_labels = load_and_preprocess_features(test_csv, feature_dir, projected)

            # Standardize the test features using the scaler fitted on training data
            test_features_scaled = scaler.transform(test_features)

            # Predict
            test_predictions = best_classifier.predict(test_features_scaled)

            # Calculate overall accuracy
            overall_acc = accuracy_score(test_labels, test_predictions)
            print(f"  Test Dataset {j}-{j+2} Overall Accuracy: {overall_acc:.4f}")

            # Calculate accuracy per distance
            csv_data = pd.read_csv(test_csv)
            dataset_overall_results["MLPClassifier"][f"Test_Dataset_{j}-{j+2}"] = float(overall_acc)

        # Save results for this trained model
        results[f"Trained_on_Dataset_{i}-{i+2}"] = dataset_results
        overall_results[f"Trained_on_Dataset_{i}-{i+2}"] = dataset_overall_results
        #misclassification_counts[f"Trained_on_Dataset_{i}-{i+2}"] = dataset_misclassification_counts

    # Save detailed results
    result_type = "after_projection" if projected else "before_projection"
    filename = f"./Phi_results/line_{save_string}/{result_type}/classification_results.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed classification results saved to {filename}")

    # Save overall results
    overall_filename = f"./Phi_results/line_{save_string}/{result_type}/overall_classification_results.json"
    with open(overall_filename, "w") as f:
        json.dump(overall_results, f, indent=2)
    print(f"Overall classification results saved to {overall_filename}")

# Process features before and after projection
print("Processing Phi features before projection...")
process_phi_features(projected=False, num_datasets=5)

print("\nProcessing Phi features after projection...")
process_phi_features(projected=True, num_datasets=5)

print("\nAll processing completed.")

