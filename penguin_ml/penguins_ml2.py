# Import the necessary libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Read the penguin data from a CSV file
penguin_df = pd.read_csv('penguins.csv')

# Drop rows with missing values (NaN)
penguin_df.dropna(inplace=True)

# Define the target variable 'output' as the 'species' column
output = penguin_df['species']

# Define the feature variables, which include various penguin characteristics
features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]

# Convert categorical features into binary (one-hot encoding)
features = pd.get_dummies(features)

# Encode the target variable 'output' into numerical labels
output, uniques = pd.factorize(output)

# Split the data into training and testing sets (80% for testing)
x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=.8)

# Create a Random Forest Classifier model with a specified random seed
rfc = RandomForestClassifier(random_state=15)

# Train (fit) the Random Forest Classifier model on the training data
rfc.fit(x_train, y_train)

# Use the trained model to make predictions on the test data
y_pred = rfc.predict(x_test)

# Calculate the accuracy of the model by comparing predicted labels to actual labels
score = accuracy_score(y_pred, y_test)

# Print the accuracy score
print('Our accuracy score for this model is {}'.format(score))
