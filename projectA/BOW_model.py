import os
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

from sklearn.calibration import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, log_loss
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
import sklearn


DATA_DIR = 'data_readinglevel'

encoder = LabelEncoder()
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def load_train_data():
    '''Load training and test data from CSV files.'''
    x_train_df = pd.read_csv(os.path.join(DATA_DIR, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv'))
    y_train_df = encoder.fit_transform(y_train_df['Coarse Label'])
    return x_train_df, y_train_df



def custom_tokenizer(text):
    # Lowercase and remove punctuation
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    # Tokenize, remove stopwords, and stem/lemmatize in one step
    return [ps.stem(lemmatizer.lemmatize(token)) for token in word_tokenize(text) if token not in stop_words]

def create_pipeline():
    preprocesser = ColumnTransformer(transformers=[
        ('text', TfidfVectorizer(
            tokenizer=custom_tokenizer,
        ), 'text')
    ])
    pipeline = Pipeline([
        ('preprocesser', preprocesser),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    return pipeline

def train_model(x_train_df=None, y_train=None, params=None):
    pipeline = create_pipeline()
    if params:
        pipeline.set_params(**params)
    pipeline.fit(x_train_df, y_train)
    return pipeline

def predict(pipeline: Pipeline, x_test_df):
    return pipeline.predict_proba(x_test_df)[:, 1]


def cross_validate_model():
    '''Perform cross-validation on the model.'''
    x_train_df, y_train_df = load_train_data()
    param_grid = {
        'classifier__C': [.01,0.1, 1, 10, 100],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['lbfgs'],
        'preprocesser__text__ngram_range': [(1, 1)],
        'preprocesser__text__max_df': [0.5, 0.75],
        'preprocesser__text__min_df': [2, 5, 10],
        'preprocesser__text__max_features': [ 5000, 10000, 20000]
    }
    
    
    pipeline = create_pipeline()
    
    stratified_kfold = StratifiedKFold(n_splits=3, shuffle=False)
    grid_search = RandomizedSearchCV(
    pipeline, param_distributions=param_grid,
    n_iter=50, cv=stratified_kfold,
    scoring='accuracy', n_jobs=-1, verbose=2
)

    grid_search.fit(x_train_df, y_train_df)
    
    print("Best params:", grid_search.best_params_)
    print("Best CV acc:", grid_search.best_score_)
    return grid_search, grid_search.best_params_

def model_test(params):
    '''Test the model using the test dataset.'''
    x_train_df, y_train_df = load_train_data()
    
    X_train, X_test, y_train, y_test = train_test_split(x_train_df, y_train_df, test_size=0.2, random_state=42)
    
    
    
    pipeline = train_model(X_train, y_train, params=params)
    
    predictions = predict(pipeline, X_test)

    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true = y_test, y_score=predictions)
    auroc_score = sklearn.metrics.roc_auc_score(y_true = y_test, y_score=predictions)
    roc_display = sklearn.metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auroc_score)
    y_pred = pipeline.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    # confusion matrix
    cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)


    
    print(f"AUROC Score: {auroc_score}")
    roc_display.plot()
    plt.show()

    return 


def plot_hyperparameter_performance():
    '''Plot training and validation performance (log loss) as a function of hyperparameter values.'''
    x_train_df, y_train_df = load_train_data()
    
    # Define the hyperparameter values to test
    hyperparameter_values = [0.01, 0.1, 1, 10, 100]
    
    # Initialize lists to store results
    train_scores = []
    val_scores = []
    train_scores_std = []
    val_scores_std = []
    
    # Perform cross-validation for each hyperparameter value
    for C in hyperparameter_values:
        pipeline = create_pipeline()
        pipeline.set_params(classifier__C=C)
        
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_fold_scores = []
        val_fold_scores = []
        
        for train_idx, val_idx in stratified_kfold.split(x_train_df, y_train_df):
            X_train, X_val = x_train_df.iloc[train_idx], x_train_df.iloc[val_idx]
            y_train, y_val = y_train_df[train_idx], y_train_df[val_idx]
            
            pipeline.fit(X_train, y_train)
            
            # Compute training and validation log loss
            train_fold_scores.append(log_loss(y_train, pipeline.predict_proba(X_train)))
            val_fold_scores.append(log_loss(y_val, pipeline.predict_proba(X_val)))
        
        # Store mean and standard deviation of scores
        train_scores.append(np.mean(train_fold_scores))
        val_scores.append(np.mean(val_fold_scores))
        train_scores_std.append(np.std(train_fold_scores))
        val_scores_std.append(np.std(val_fold_scores))
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.errorbar(hyperparameter_values, train_scores, yerr=train_scores_std, label='Training Log Loss', fmt='-o', capsize=5)
    plt.errorbar(hyperparameter_values, val_scores, yerr=val_scores_std, label='Validation Log Loss', fmt='-o', capsize=5)
    
    # Add labels, legend, and title
    plt.xscale('log')  # Use a logarithmic scale for C
    plt.xlabel('Regularization Strength (C)')
    plt.ylabel('Log Loss')
    plt.title('Training and Validation Log Loss vs. Regularization Strength')
    plt.legend()
    plt.grid(True)
    plt.show()

def model_predict(params: dict):
    x_test_df = pd.read_csv(os.path.join(DATA_DIR, 'x_test.csv'))
    x_train_df, y_train_df = load_train_data()

    model = train_model(x_train_df=x_train_df, y_train=y_train_df, params=params)

    predictions = predict(model, x_test_df)
    filename = "yproba1_test.txt"

    # Save predictions to a file
    with open(filename, 'w') as f:
        for prediction in predictions:
            f.write(f"{prediction}\n")







if __name__ == '__main__':
    # best_model ,best_params = cross_validate_model()
    # best_params =  {'preprocesser__text__ngram_range': (1, 1), 'preprocesser__text__min_df': 2, 'preprocesser__text__max_features': 10000, 'preprocesser__text__max_df': 0.5, 'classifier__solver': 'liblinear', 'classifier__penalty': 'l2', 'classifier__C': 0.1}
    # model_test(best_params)
    # model_predict(best_params)
    # # Call the function to generate the plot
    plot_hyperparameter_performance()


    