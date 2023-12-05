from ucimlrepo import fetch_ucirepo
import pandas as pd

# Fetch the Iris dataset
iris = fetch_ucirepo(id=53)

# Extract the features and target into separate DataFrames
X = iris.data.features  # Predictor variables
y = iris.data.targets  # Response variables

# Combine features and targets into one DataFrame
iris_df = pd.concat([X, y], axis=1)

# Rename the columns for consistency and ease of use
iris_df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Now you can perform species-specific linear regressions as shown previously
models = {}

# Iterate over each species to create separate models
for species in iris_df['species'].unique():
    # Subset the DataFrame by species
    subset = iris_df[iris_df['species'] == species]

    # Extract the petal width and length for the current species
    X = subset['petal_width'].values
    y = subset['petal_length'].values

    # Calculate the means
    x_mean = sum(X) / len(X)
    y_mean = sum(y) / len(y)

    # Calculate the numerator and denominator for the slope (beta_1)
    numerator = sum((X - x_mean) * (y - y_mean))
    denominator = sum((X - x_mean) ** 2)

    # Calculate slope (beta_1) and intercept (beta_0)
    beta_1 = numerator / denominator
    beta_0 = y_mean - beta_1 * x_mean

    # Store the model coefficients for the current species
    models[species] = (beta_0, beta_1)

# Output the coefficients for each species
for species, (beta_0, beta_1) in models.items():
    print(f"{species} - Intercept (β0): {beta_0}, Slope (β1): {beta_1}")




import matplotlib.pyplot as plt
import numpy as np


# Assuming 'iris_df' is a pandas DataFrame containing the iris dataset,
# with columns ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Calculate the overall regression line for all species
X_all = iris_df['petal_width'].values
y_all = iris_df['petal_length'].values

# Calculate the means
x_all_mean = np.mean(X_all)
y_all_mean = np.mean(y_all)

# Calculate the numerator and denominator for the overall slope (beta_1)
numerator_all = np.sum((X_all - x_all_mean) * (y_all - y_all_mean))
denominator_all = np.sum((X_all - x_all_mean) ** 2)

# Calculate overall slope (beta_1) and intercept (beta_0)
beta_1_all = numerator_all / denominator_all
beta_0_all = y_all_mean - beta_1_all * x_all_mean

# Plot the scatter and regression lines
plt.figure(figsize=(10, 6))

# Scatter plot for each species
species_list = iris_df['species'].unique()
colors = ['blue', 'green', 'orange']
for species, color in zip(species_list, colors):
    subset = iris_df[iris_df['species'] == species]
    plt.scatter(subset['petal_width'], subset['petal_length'], label=species, alpha=0.5)

# Regression line for each species
for species, color in zip(species_list, colors):
    subset = iris_df[iris_df['species'] == species]
    X = subset['petal_width'].values
    y = subset['petal_length'].values
    # Same calculations for slope and intercept as before, using 'X' and 'y' for each species
    # ...
    # Here we use the models dictionary from the previous snippet
    beta_0, beta_1 = models[species]
    # Plot the regression line
    plt.plot(X, beta_0 + beta_1 * X, color=color, linestyle='dashed')

# Overall regression line
plt.plot(X_all, beta_0_all + beta_1_all * X_all, color='red', linestyle='dashed', label='Overall')

plt.xlabel('Petal Width')
plt.ylabel('Petal Length')
plt.title('Iris Petal Width vs Length with Regression Lines')
plt.legend()
plt.show()
