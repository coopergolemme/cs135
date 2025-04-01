import BOW_model

def test_load_data():
    '''Test the load_train_test_data function.'''

    x_train_df, y_train_df = BOW_model.load_train_data()
    N, n_cols = x_train_df.shape
    P, n_cols_y = y_train_df.shape

    assert N == 5557
    assert P == 5557


    
def test_vectorize_text():
    '''Test the vectorize_text function.'''

    x_train_df, y_train_df = BOW_model.load_train_data()
    X = x_train_df['text'].values.tolist()

    X_features = BOW_model.vectorize_text(X)

    assert X_features.shape[0] == 5557

def test_train_model():
    '''Test the train_model function.'''
    BOW_model.train_model()
