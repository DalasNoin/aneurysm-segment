import unittest
import preprocessing
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_something(self):
        test_class = preprocessing.AneurysmData(path = None, mock=True)
        test_dims = np.random.randint(1, 100, size=(50, 2))
        test_images = np.random.randint(1, 100, size=(50, 100, 100))
        for i1 in range(test_dims.shape[0]):
            shape = test_dims[i1]
            shape_real = test_class.downsample_image(test_images[i1], size=(shape)).shape
            assert (shape == shape_real).all()


if __name__ == '__main__':
    unittest.main()
