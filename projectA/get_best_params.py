import joblib

# Load the model from the file
best_model = joblib.load('/Users/coopergolemme/tufts/cs/cs135/projectA/best_model_2.pkl')

# Get the model parameters
model_params = best_model.get_params()

# Print the parameters
print("Model Parameters:")
print(model_params)
