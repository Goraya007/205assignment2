
from ucimlrepo import fetch_ucirepo
import pandas as pd
# Fetch the Iris dataset
iris = fetch_ucirepo(id=53)
# Assuming iris_df is a pandas DataFrame with the Iris dataset already loaded
# Extract the features and target into separate DataFrames
X = iris.data.features  # Predictor variables
y = iris.data.targets  # Response variables

# Combine features and targets into one DataFrame
iris_df = pd.concat([X, y], axis=1)
# Calculate descriptive statistics for the continuous variables
descriptive_stats = iris_df.describe()

# The 'describe' method returns the following statistics by default:
# count, mean, std (standard deviation), min, 25% (Q1), 50% (median, Q2), 75% (Q3), max

# If you want to format the output to match the image exactly, you can proceed as follows:
formatted_stats = descriptive_stats.loc[['min', '25%', '50%', 'mean', '75%', 'max']]
formatted_stats.index = ['Min.', '1st Qu.', 'Median', 'Mean', '3rd Qu.', 'Max.']  # Rename the index

# Print or output the formatted descriptive statistics
print(formatted_stats)
