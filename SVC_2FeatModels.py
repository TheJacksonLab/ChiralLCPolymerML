import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem

# Function to compute features
def compute_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Add explicit hydrogens
    mol = Chem.AddHs(mol)
    
    # Generate 3D conformer
    try:
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)
    except:
        return None  # If embedding fails, skip this molecule

    features = {
        'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings(mol),
        'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings(mol),
        'NumSpiroAtoms': rdMolDescriptors.CalcNumSpiroAtoms(mol),
        'NumBridgeheadAtoms': rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        'MolLogP': Descriptors.MolLogP(mol),
        'MolWt': Descriptors.MolWt(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'TPSA': Descriptors.TPSA(mol),
        'InertialShapeFactor': rdMolDescriptors.CalcInertialShapeFactor(mol),
        'Eccentricity': rdMolDescriptors.CalcEccentricity(mol),
        'Asphericity': rdMolDescriptors.CalcAsphericity(mol)
    }
    return features

# Read the Data
df = pd.read_csv("../../Basic_Params.csv")
molecs = 41
data_labels = np.array(df.columns.values[3:])
all_features = np.array(df.iloc[:molecs, 3:])
target = np.array(df.iloc[:molecs, 2])

# Compute features for each molecule in the dataset
features_list = []
for smiles in df['SMILES']:  # Assuming the SMILES strings are in a column named 'SMILES'
    features = compute_features(smiles)
    if features is not None:
        features_list.append(features)

# Convert the features list to a DataFrame
features_df = pd.DataFrame(features_list)

# Append the new features to the original dataset
df = pd.concat([df, features_df], axis=1)

# Remove any columns with names starting with 'Unnamed'
df = df.loc[:, ~df.columns.str.contains('Unnamed')]

# Update all_features to include RDKit features
new_features = np.array(df.iloc[:, 36:])
all_features = np.hstack((all_features, new_features))

# Update data_labels to include RDKit features
rdkit_feature_names = features_df.columns.values
data_labels = np.concatenate((data_labels, rdkit_feature_names))

# Replace NaN values with 0 for general features
all_features = np.nan_to_num(all_features, nan=0.0)

# Define the features to check for NaN values
features_to_check = ['Mn', 'Mw', 'PDI']

# Check if features_to_check exist in data_labels
indices_to_check = [np.where(data_labels == feature)[0][0] for feature in features_to_check if feature in data_labels]

# Exclude rows where Mn, Mw, or PDI are NaN
if indices_to_check:
    mask = np.all(~np.isnan(all_features[:, indices_to_check]), axis=1)
    all_features = all_features[mask]
    target = target[mask]

# Scale X to unit variance and zero mean
st = StandardScaler()
X = st.fit_transform(all_features)
feats = X.shape[1]
y = target

# Regularization parameters
C_reg = np.array([5, 10, 30, 100, 300, 1000])  # reg. strength = 1/C_reg
C_regs = C_reg.shape[0]
accuracies = np.zeros(C_regs)
top_y_pred = np.zeros((feats**2, molecs))
top_C_reg = np.zeros(feats**2)
top_accuracies = np.zeros(feats**2)
out = "FeatA,FeatB,Accuracy,Creg\n"
All_accuracies = np.zeros((feats, feats), dtype=float)

# Leave-One-Out Cross-Validation (LOOV)
loo = LeaveOneOut()

# Iterate over feature pairs
for k in range(feats):
    for l in range(feats):
        if l <= k:  # Skip redundant pairs
            continue
        y_pred = np.zeros((C_regs, molecs))  # Adjusted here
        del_feats = np.arange(0, feats, dtype=int)
        
        del_feats = np.delete(del_feats, [k, l], 0)
        
        X_temp = np.delete(X, del_feats, 1)  # only selected features
        
        for j in range(C_regs):
            accuracies_fold = np.zeros(molecs)  # To store accuracy for each fold
            for i, (train_index, test_index) in enumerate(loo.split(X_temp)):
                X_train, X_test = X_temp[train_index], X_temp[test_index]
                y_train, y_test = y[train_index], y[test_index]

                svc = SVC(kernel="rbf", C=C_reg[j])
                svc.fit(X_train, y_train)
                y_pred[j, i] = svc.predict(X_test)

            accuracies[j] = accuracy_score(y, y_pred[j])

        # Find best C parameter
        bestC = np.argmax(accuracies)

        top_y_pred[k * feats + l] = y_pred[bestC]
        top_C_reg[k * feats + l] = C_reg[bestC]
        top_accuracies[k * feats + l] = accuracies[bestC]
        All_accuracies[k, l] = accuracies[bestC]

        print("{},{}, Accuracy={:.2f}, Creg={:.0f}".format(data_labels[k], data_labels[l], accuracies[bestC], C_reg[bestC]))
        out += "{},{},{:.4f},{:.0f}\n".format(data_labels[k], data_labels[l], accuracies[bestC], C_reg[bestC])

# Combine (featA, featB) and (featB, featA) and keep unique pairs
unique_pairs = {}
for k in range(feats):
    for l in range(k + 1, feats):  # Ensure l > k to avoid duplicates
        accuracy = All_accuracies[k, l]
        pair = tuple(sorted([data_labels[k], data_labels[l]]))
        if pair not in unique_pairs or unique_pairs[pair] < accuracy:
            unique_pairs[pair] = accuracy

# Get the 10 best unique pairs
sorted_pairs = sorted(unique_pairs.items(), key=lambda x: x[1], reverse=True)[:10]

# Print the 10 best pairs with accuracy
for pair, accuracy in sorted_pairs:
    print(f"Pair: {pair}, Accuracy: {accuracy:.4f}")

# Get the best feature pair index for plotting
best_pair = sorted_pairs[0][0]
bestF = np.where((data_labels == best_pair[0]) | (data_labels == best_pair[1]))[0]

# Save performance results
with open("FeaturePerformance2.csv", 'w') as fileo:
    fileo.write(out)

# Generate and save heatmap or other visualizations as needed
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.sans-serif'] = "Arial"
fig, ax = plt.subplots(figsize=(8, 8))  # Make the figure larger
ax.scatter(y, top_y_pred[bestF[0] * feats + bestF[1]], color='mediumvioletred')
ax.plot([0, 1], [0, 1], color='gray')  # Use [0, 1] range for accuracy
ax.annotate("Accuracy = {:.2f}".format(top_accuracies[bestF[0] * feats + bestF[1]]), xy=(0.1, 0.9), size=15)
ax.annotate("C = {:.0f}".format(top_C_reg[bestF[0] * feats + bestF[1]]), xy=(0.1, 0.8), size=15)
ax.set_ylabel("Predicted target", fontsize='x-large')
ax.set_xlabel("Actual target", fontsize='x-large')
fig.tight_layout()
fig.savefig("TopFeaturePerformance2_{}.png".format(bestF[0] * feats + bestF[1]))
plt.close()

