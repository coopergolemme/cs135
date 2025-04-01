import joblib
import pandas as pd
import numpy as np
from load_BERT_embeddings import load_arr_from_npz
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer



best_model = joblib.load('/Users/coopergolemme/tufts/cs/cs135/projectA/best_model_2_whole_set.pkl')



print(best_model)

# Load the test data

DATA_DIR = "data_readinglevel/"


test_text = pd.read_csv(DATA_DIR + "X_test.csv")['text']
X_numerical_test = pd.read_csv(DATA_DIR + "X_test.csv").iloc[:, 6:]
bert_embeds_test = load_arr_from_npz(DATA_DIR + "x_test_BERT_embeddings.npz")

# Preprocess BERT embeddings (PCA for dimensionality reduction)
pca_bert = TruncatedSVD(n_components=100)
bert_reduced = pca_bert.fit_transform(bert_embeds_test)
scaler_bert = StandardScaler()
bert_scaled = scaler_bert.fit_transform(bert_reduced)

# Preprocess numerical features
imputer = SimpleImputer(strategy='mean')
X_num = imputer.fit_transform(X_numerical_test)
scaler_num = StandardScaler()
num_scaled = scaler_num.fit_transform(X_num)

# Preprocess text with TF-IDF + SVD
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 1))
X_tfidf = tfidf.fit_transform(test_text)
svd = TruncatedSVD(n_components=300)
tfidf_reduced = svd.fit_transform(X_tfidf)
scaler_tfidf = StandardScaler()
tfidf_scaled = scaler_tfidf.fit_transform(tfidf_reduced)

# Combine all features
X_combined = np.hstack([bert_scaled, num_scaled, tfidf_scaled])
# Make predictions
predictions = best_model.predict_proba(X_combined)[:, 1]  # Probability of the positive class
# Save predictions to CSV
print(predictions)

# write to a text file
with open('predictions.txt', 'w') as f:
    for item in predictions:
        f.write("%s\n" % item)

