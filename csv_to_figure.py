import pandas as pd
import matplotlib.pyplot as plt

# Load CSV and round floats to 2 decimals without affecting integers
df = pd.read_csv('tables/fid_scores.csv').applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x)

# Plot and save as figure
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

plt.savefig('table_figure.png', bbox_inches='tight', dpi=300)
plt.close()