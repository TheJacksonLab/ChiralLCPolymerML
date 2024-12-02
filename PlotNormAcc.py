import matplotlib.pyplot as plt
import numpy as np
import csv

# Read the data file
file_path = 'AllFeaturePerformance4_withConstant.csv'
with open(file_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    data = list(reader)

# Extract features and accuracy values
feature_dict = {}

for row in data:
    features = row[:4]  # Collect all four features in the model (ConstantFeat, FeatB, FeatC, FeatD)
    accuracy = float(row[4])

    # Add the accuracy value to each feature's list
    for feature in features:
        feature = feature.strip()  # Clean up the feature name
        if feature not in feature_dict:
            feature_dict[feature] = []
        feature_dict[feature].append(accuracy)

# List of features to plot histograms for
features_to_plot = [
    '# O atoms on SC', 'Mn', 'NumSaturatedRings', 'NumSpiroAtoms', 'NumBridgeheadAtoms',
    'NumHDonors', '# C atoms on SC', '# of heteroatoms atoms on BB', 'TPSA', 
    'Ratom_C', 'Rring_2', '# atom after braching', 
    '# atoms (C and O) between braching and BB', 'Ratom_all'
]

# Filter feature_dict to only include specified features
filtered_feature_dict = {feature: accuracies for feature, accuracies in feature_dict.items() if feature in features_to_plot}

# Compute the total number of models per feature
total_models = {feature: len(accuracies) for feature, accuracies in filtered_feature_dict.items()}

# Compute the sum of accuracy values and normalize
feature_sums = {feature: np.sum(np.array(accuracies)) for feature, accuracies in filtered_feature_dict.items()}
normalized_sums = {feature: feature_sums[feature] / total_models[feature] for feature in feature_sums}

# Sort features by normalized sum value (optional, but makes the bar plot more organized)
sorted_features = sorted(normalized_sums.items(), key=lambda x: x[1], reverse=True)

# Separate features and their normalized sums for plotting
features, normalized_sums_values = zip(*sorted_features)

# Print raw x (features) and y (normalized sums) values
for feature, norm_sum in zip(features, normalized_sums_values):
    print(f"Feature: {feature}, Normalized Accuracy: {norm_sum:.4f}")

# Plot settings
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams.update({'font.size': 14})  # Set base font size
fig, ax = plt.subplots(figsize=(10, 8), tight_layout=True)

# Create a bar plot with dark color (dark blue)
bars = ax.bar(features, normalized_sums_values, color='darkblue')

# Rotate x-axis labels for better readability
ax.set_xticklabels(features, rotation=45, ha='right', fontsize=14)

# Add labels and title with larger font sizes
ax.set_xlabel('Features', fontsize=16)
ax.set_ylabel('Normalized Accuracy', fontsize=16)
ax.set_title('Normalized Accuracy Values for Selected Features', fontsize=18)

# Set y-axis limits
ax.set_ylim(0.5, 0.9)

# Show plot
plt.show()

# Optionally save the plot
# plt.savefig('normalized_accuracy_bar_plot.png', dpi=600)

