import pandas as pd
import numpy as np
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
df = pd.read_csv("../../../Basic_Params.csv")
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

# Scale X to unit variance and zero mean
st = StandardScaler()
X = st.fit_transform(all_features)
feats = X.shape[1]
y = target

# Regularization parameters
C_reg = np.array([30, 100, 300, 1000])  # reg. strength = 1/C_reg
C_regs = C_reg.shape[0]
accuracies = np.zeros(C_regs)
top_results = []

# Leave-One-Out Cross-Validation (LOOV)
loo = LeaveOneOut()

# Iterate over all possible constant features
for const_feat in range(feats):
    print(f"Processing constant feature {const_feat + 1}/{feats} ({data_labels[const_feat]})...")
    
    # Iterate over combinations of the remaining three features
    for k in range(feats):
        if k == const_feat:
            continue
        for l in range(k + 1, feats):
            if l == const_feat:
                continue
            for m in range(l + 1, feats):
                if m == const_feat:
                    continue

                print(f"  Testing combination: {data_labels[const_feat]}, {data_labels[k]}, {data_labels[l]}, {data_labels[m]}")

                y_pred = np.zeros((C_regs, molecs))
                del_feats = np.arange(0, feats, dtype=int)
                del_feats = np.delete(del_feats, [const_feat, k, l, m], 0)

                X_temp = np.delete(X, del_feats, 1)  # Select the constant feature and three others

                for j in range(C_regs):
                    print(f"    Testing regularization parameter C = {C_reg[j]}")

                    for i, (train_index, test_index) in enumerate(loo.split(X_temp)):
                        X_train, X_test = X_temp[train_index], X_temp[test_index]
                        y_train, y_test = y[train_index], y[test_index]

                        svc = SVC(kernel="rbf", C=C_reg[j])
                        svc.fit(X_train, y_train)
                        y_pred[j, i] = svc.predict(X_test)

                    accuracies[j] = accuracy_score(y, y_pred[j])

                # Find best C parameter
                bestC = np.argmax(accuracies)
                print(f"      Best accuracy: {accuracies[bestC]:.4f} with C = {C_reg[bestC]}")

                top_results.append((
                    data_labels[const_feat],
                    data_labels[k],
                    data_labels[l],
                    data_labels[m],
                    accuracies[bestC],
                    C_reg[bestC]
                ))

# Sort results by accuracy and keep the top 10
top_results = sorted(top_results, key=lambda x: x[4], reverse=True)[:10]

# Save the top 10 results
out = "ConstantFeat,FeatB,FeatC,FeatD,Accuracy,Creg\n"
for result in top_results:
    out += "{},{},{},{},{:.4f},{:.0f}\n".format(*result)

with open("FeaturePerformance4_withConstant.csv", 'w') as fileo:
    fileo.write(out)

# Print the best result
best_result = top_results[0]
print("Best Model is {}, {}, {}, {}, Accuracy = {:.2f}".format(*best_result[:4], best_result[4]))

