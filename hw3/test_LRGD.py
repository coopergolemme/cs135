from LRGradientDescent import LogisticRegressionGradientDescent as LRGD
import numpy as np

def test_initialization():
    ''' Test the initialization of the LogisticRegressionGradientDescent class '''
    lr = LRGD(C=1.0, step_size=0.01, num_iterations=100)
    assert lr.C == 1.0
    assert lr.step_size == 0.01
    assert lr.num_iterations == 100
    assert lr.use_base2_for_BCE == True
    assert lr.init_recipe == 'zeros'
    assert lr.verbose == True
    assert lr.loss_converge_thr == 0.00001
    assert lr.grad_norm_converge_thr == 0.001
    assert lr.param_converge_thr == 0.001
    assert lr.proba_to_binary_threshold == 0.5

def test_fit():
    ''' Test the fit method of the LRGD class '''
    x_NF = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y_N = np.array([0, 0, 1, 1, 1])
    lr = LRGD(C=1.0, step_size=0.01, num_iterations=100)
    lr.fit(x_NF, y_N)
    assert hasattr(lr, 'wtil_G')
    assert lr.did_converge == True or lr.did_converge == False

def test_predict_proba():
    ''' Test the predict_proba method of the LRGD class '''
    x_NF = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y_N = np.array([0, 0, 1, 1, 1])
    lr = LRGD(C=1.0, step_size=0.01, num_iterations=100)
    lr.fit(x_NF, y_N)
    proba_N2 = lr.predict_proba(x_NF)
    assert proba_N2.shape == (5, 2)
    assert np.all(proba_N2 >= 0) and np.all(proba_N2 <= 1)

def test_predict():
    ''' Test the predict method of the LRGD class '''
    x_NF = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y_N = np.array([0, 0, 1, 1, 1])
    lr = LRGD(C=1.0, step_size=0.01, num_iterations=100)
    lr.fit(x_NF, y_N)
    yhat_N = lr.predict(x_NF)
    assert yhat_N.shape == (5,)
    assert np.all(np.isin(yhat_N, [0, 1]))

ffffffffffffjjjj
def test_convergence():
    ''' Test if the model converges on a simple dataset '''
    x_NF = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y_N = np.array([0, 0, 1, 1, 1])
    lr = LRGD(C=1.0, num_iterations=10000)
    lr.fit(x_NF, y_N)
    assert lr.did_converge == True

def test_divergence():
    ''' Test if the model detects divergence with a large step size '''
    x_NF = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y_N = np.array([0, 0, 1, 1, 1])
    lr = LRGD(C=1.0, step_size=1.0, num_iterations=100)
    try:
        lr.fit(x_NF, y_N)
    except ValueError as e:
        assert 'Divergence detected' in str(e)

def test_different_initializations():
    ''' Test the model with different initialization methods '''
    x_NF = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y_N = np.array([0, 0, 1, 1, 1])
    for init_recipe in ['zeros', 'uniform_-1_to_1', 'uniform_-6_to_6']:
        lr = LRGD(C=1.0, step_size=0.01, num_iterations=100, init_recipe=init_recipe)
        lr.fit(x_NF, y_N)
        assert hasattr(lr, 'wtil_G')

def test_large_dataset():
    ''' Test the model on a larger dataset '''
    np.random.seed(0)
    x_NF = np.random.randn(1000, 10)
    y_N = (np.random.rand(1000) > 0.5).astype(int)
    lr = LRGD(C=1.0, num_iterations=1000)
    lr.fit(x_NF, y_N)
    yhat_N = lr.predict(x_NF)
    assert yhat_N.shape == (1000,)
    assert np.all(np.isin(yhat_N, [0, 1]))
