import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
from load_BERT_embeddings import load_arr_from_npz
from scipy.stats import loguniform, uniform

DATA_DIR = "data_readinglevel/"


# Encode the labels
encoder = LabelEncoder()

# Load the training data
X_train_text = pd.read_csv(DATA_DIR + "X_train.csv")['text']
X_train_numerical = pd.read_csv(DATA_DIR + "X_train.csv").iloc[:, 6:]
Y_train = pd.read_csv(DATA_DIR + "y_train.csv")
Y_train = encoder.fit_transform(Y_train['Coarse Label'])
X_train_bert_embeds = load_arr_from_npz(DATA_DIR + "x_train_BERT_embeddings.npz")

# Combine all features into a single DataFrame
bert_columns = [f'bert_{i}' for i in range(X_train_bert_embeds.shape[1])]
bert_df = pd.DataFrame(X_train_bert_embeds, columns=bert_columns)
X_train = pd.concat([X_train_text, X_train_numerical, bert_df], axis=1)

def create_pipeline(text_col: str,
                    num_cols: list[str],
                    bert_cols: list[str]) -> Pipeline:
    # Text preprocessing pipeline
    text_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 1))),
        ('svd', TruncatedSVD(n_components=300)),
        ('scaler', StandardScaler())
    ])

    # Numerical feature pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # BERT embedding pipeline
    bert_pipeline = Pipeline([
        ('pca', TruncatedSVD(n_components=100)),
        ('scaler', StandardScaler())
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer([
        ('text', text_pipeline, text_col),
        ('num', num_pipeline, num_cols),
        ('bert', bert_pipeline, bert_cols)
    ])

    # Full pipeline with XGBoost classifier
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(tree_method='hist', enable_categorical=False)) 
    ])
    return full_pipeline


# Define column groups
# text_col = 'text'                   
# num_cols = X_train_numerical.columns.tolist()
# bert_cols = bert_columns

# # Create the pipeline
# full_pipeline = create_pipeline(text_col, num_cols, bert_cols)

# Load the best model from a file
best_model = joblib.load('best_model_2.pkl')

# print(full_pipeline)

# train on the full dataset
best_model.fit(X_train, Y_train)


# # Hyperparameter search space for SVM
# param_dist = {
#     'classifier__C': loguniform(1e-2, 1e3),  # Regularization strength
#     'classifier__gamma': loguniform(1e-4, 1e1),  # Kernel bandwidth
#     'classifier__kernel': ['rbf', 'linear'],  # Test both kernels
#     'preprocessor__text__svd__n_components': [100, 200, 300]  # Optional: tune SVD
# }

# Randomized search
# random_search = RandomizedSearchCV(
#     full_pipeline,
#     param_distributions=param_dist,
#     n_iter=10,
#     cv=3,
#     scoring='accuracy',
#     verbose=2,
#     n_jobs=-1
# )

# random_search.fit(X_train, Y_train)

# # Save pipeline
# joblib.dump(random_search.best_estimator_, 'svm_text_classifier.joblib')