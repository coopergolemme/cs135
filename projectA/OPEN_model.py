import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import log_loss, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
from load_BERT_embeddings import load_arr_from_npz
import joblib



DATA_DIR = "data_readinglevel/"


# Encode the labels
encoder = LabelEncoder()

# Load the training data
train_text = pd.read_csv(DATA_DIR + "X_train.csv")['text']
X_numerical = pd.read_csv(DATA_DIR + "X_train.csv").iloc[:, 6:]
y_train = pd.read_csv(DATA_DIR + "y_train.csv")
y_train = encoder.fit_transform(y_train['Coarse Label'])
bert_embeds = load_arr_from_npz(DATA_DIR + "x_train_BERT_embeddings.npz")



# Preprocess BERT embeddings (PCA for dimensionality reduction)
pca_bert = TruncatedSVD(n_components=100)
bert_reduced = pca_bert.fit_transform(bert_embeds)
scaler_bert = StandardScaler()
bert_scaled = scaler_bert.fit_transform(bert_reduced)

# Preprocess numerical features
imputer = SimpleImputer(strategy='mean')
X_num = imputer.fit_transform(X_numerical)
scaler_num = StandardScaler()
num_scaled = scaler_num.fit_transform(X_num)

# Preprocess text with TF-IDF + SVD
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 1))
X_tfidf = tfidf.fit_transform(train_text)
svd = TruncatedSVD(n_components=300)
tfidf_reduced = svd.fit_transform(X_tfidf)
scaler_tfidf = StandardScaler()
tfidf_scaled = scaler_tfidf.fit_transform(tfidf_reduced)

# Combine all features
X_combined = np.hstack([bert_scaled, num_scaled, tfidf_scaled])



def plot_max_depth_performance(best_params):
    '''Plot training and validation performance as a function of max_depth.'''
    model = XGBClassifier(**best_params)

    # Define the hyperparameter values to test
    max_depth_values = [2, 4, 6, 8, 10]
    
    # Initialize lists to store results
    train_scores = []
    val_scores = []
    train_scores_std = []
    val_scores_std = []
    
    # Perform cross-validation for each max_depth value
    for max_depth in max_depth_values:
        model = XGBClassifier(max_depth=max_depth, **best_params)
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_fold_scores = []
        val_fold_scores = []
        
        for train_idx, val_idx in stratified_kfold.split(X_combined, y_train):
            X_train, X_val = X_combined[train_idx], X_combined[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
            
            model.fit(X_train, y_train_fold)
            
            # Compute training and validation log loss
            train_fold_scores.append(log_loss(y_train_fold, model.predict_proba(X_train)))
            val_fold_scores.append(log_loss(y_val_fold, model.predict_proba(X_val)))
        
        # Store mean and standard deviation of scores
        train_scores.append(np.mean(train_fold_scores))
        val_scores.append(np.mean(val_fold_scores))
        train_scores_std.append(np.std(train_fold_scores))
        val_scores_std.append(np.std(val_fold_scores))
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.errorbar(max_depth_values, train_scores, yerr=train_scores_std, label='Training Log Loss', fmt='-o', capsize=5)
    plt.errorbar(max_depth_values, val_scores, yerr=val_scores_std, label='Validation Log Loss', fmt='-o', capsize=5)
    
    # Add labels, legend, and title
    plt.xlabel('Max Depth')
    plt.ylabel('Log Loss')
    plt.title('Training and Validation Log Loss vs. Max Depth')
    plt.legend()
    plt.grid(True)
    plt.show()


def cross_validation():
    model = XGBClassifier()

    param_dist = {
        'n_estimators': randint(200, 1000),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.29),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'reg_lambda': uniform(0, 1)
    }

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=50,
        cv=3,
        scoring='roc_auc',
        verbose=2,
        n_jobs=-1
    )


    random_search.fit(X_combined, y_train)
    print("Best parameters found: ", random_search.best_params_)
    print("Best cross-validation score: ", random_search.best_score_)
    return random_search


def display_confusion_matrix(params):
    X_train, X_test, y_train_split, y_test = train_test_split(X_combined, y_train, test_size=0.2, random_state=42, stratify=y_train)


    model = XGBClassifier(**params)
    model.fit(X_train, y_train_split)

    y_pred = model.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


def train_and_save(random_search):

    # train model on the full dataset
    best_model = random_search.best_estimator_
    best_model.fit(X_combined, y_train)


    # Save the best model
    joblib.dump(best_model, 'best_model_2_whole_set.pkl')


if __name__ == "__main__":
    best_params = {'objective': 'binary:logistic', 'base_score': None, 'booster': None, 'callbacks': None, 'colsample_bylevel': None, 'colsample_bynode': None, 'colsample_bytree': 0.7652162991174537, 'device': None, 'early_stopping_rounds': None, 'enable_categorical': False, 'eval_metric': None, 'feature_types': None, 'gamma': None, 'grow_policy': None, 'importance_type': None, 'interaction_constraints': None, 'learning_rate': 0.01778594369636227, 'max_bin': None, 'max_cat_threshold': None, 'max_cat_to_onehot': None, 'max_delta_step': None, 'max_leaves': None, 'min_child_weight': None, 'monotone_constraints': None, 'multi_strategy': None, 'n_estimators': 491, 'n_jobs': None, 'num_parallel_tree': None, 'random_state': None, 'reg_alpha': None, 'reg_lambda': 0.43813152135784816, 'sampling_method': None, 'scale_pos_weight': None, 'subsample': 0.7024470436897973, 'tree_method': 'hist', 'validate_parameters': None, 'verbosity': None}
    display_confusion_matrix(best_params)
    # random_search = cross_validation()
    # train_and_save(random_search)
    # plot_max_depth_performance(best_params)