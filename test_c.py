import numpy as np

def test_shape():
    a = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    assert a.shape == (2, 3)