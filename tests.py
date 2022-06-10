from select import select
from SOM import *
import unittest


class Test(unittest.TestCase):

    def test_centroid(self):
        # self.assertRaises(centroid(-1, 2, (1,1)))
        # self.assertRaises(centroid(0, 0, (1,1)))
        c = centroid(1, np.zeros(2))
        self.assertEqual(c.id, 1)
        self.assertEqual(len(c.loc), 2)
        self.assertEqual(c.loc[0], 0)
        self.assertEqual(c.loc[1], 0)


class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.som = SOM((10, 10), 0.7, 2, np.zeros(2))

    def test_init(cls):
        cls.assertEqual(len(cls.som.cen), 100)
        cls.assertEqual(cls.som.learning_rate, 0.7)



if __name__ == '__main__':
    unittest.main()