from ucimlrepo import fetch_ucirepo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Fetch the Iris dataset
iris = fetch_ucirepo(id=53)

# Data (as pandas dataframes)
X = iris.data.features
y = iris.data.targets

# Combine features and target labels into one DataFrame
iris_df = pd.concat([X, y], axis=1)

# Set the style of seaborn
sns.set(style="ticks", color_codes=True)

# Create a pairplot of the iris data and color by class
pairplot = sns.pairplot(iris_df, hue='class')

# Enhancing the plot for better readability
pairplot.fig.suptitle("Iris Data", y=1.02)  # Title and a slight adjustment for the title to not overlap the plots

# Show the plot
plt.show()
