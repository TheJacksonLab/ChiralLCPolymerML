import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("../../Basic_Params.csv") # input data from experiments
labels = np.array(df.columns.values[3:])
# add columns for computed data
new_features = [ 
    'NumAromaticRings', 'NumSaturatedRings', 'NumAliphaticRings',
    'NumSpiroAtoms', 'NumBridgeheadAtoms', 'MolLogP', 'MolWt',
    'NumHAcceptors', 'NumHDonors', 'TPSA', 'InertialShapeFactor',
    'Eccentricity', 'Asphericity'
]

labels = np.concatenate((labels, new_features))

# Load accuracy data
accuracies = np.load('All_accuracies.npy')
accuracies = np.flip(accuracies, axis=0)

# Flip the labels for the y-axis
flipped_labels = np.flip(labels)

# Plot settings
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.sans-serif'] = "Arial"

fig, ax = plt.subplots(figsize=(15, 10))
im = ax.imshow(accuracies, cmap='inferno', vmin=0.6, vmax=0.9)  # Adjust color range

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(flipped_labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(flipped_labels)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add color bar
cbar = fig.colorbar(im)
cbar.set_label("Average LOO Validation Accuracy", size=13)

fig.tight_layout()
plt.savefig('2FeatAccuracyHeatmap.png', dpi=300)
plt.savefig('2FeatAccuracyHeatmap.eps', format='eps')
plt.show()

