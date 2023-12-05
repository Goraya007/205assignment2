import pandas as pd  # Ensure Pandas is imported
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

# Fetch the Iris dataset
iris = fetch_ucirepo(id=53)

# Data (as pandas dataframes)
X = iris.data.features
y = iris.data.targets

# Combine features and target labels into one DataFrame
iris_df = pd.concat([X, y], axis=1)

# Set the style of seaborn
sns.set(style="whitegrid")

# Define a color palette
palette = sns.color_palette("husl", 3)  # 3 colors for the 3 species

# Create a figure with a subplot for each feature
fig, axes = plt.subplots(2, 2, figsize=(12, 12))  # Adjusted for a 2x2 grid

# Generate boxplots with the same color for each species across different graphs
sns.boxplot(ax=axes[0, 0], x='class', y='petal length', data=iris_df, palette=palette)
axes[0, 0].set_title('Petal Length')

sns.boxplot(ax=axes[0, 1], x='class', y='petal width', data=iris_df, palette=palette)
axes[0, 1].set_title('Petal Width')

sns.boxplot(ax=axes[1, 0], x='class', y='sepal length', data=iris_df, palette=palette)
axes[1, 0].set_title('Sepal Length')

sns.boxplot(ax=axes[1, 1], x='class', y='sepal width', data=iris_df, palette=palette)
axes[1, 1].set_title('Sepal Width')

# Set the titles and labels
for i in range(2):
    for j in range(2):
        axes[i, j].set_xlabel('Iris Species')
        axes[i, j].set_ylabel('')

# Enhancing the plot for better readability
plt.tight_layout()

# Show the plot
plt.show()
