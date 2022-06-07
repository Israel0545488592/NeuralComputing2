from select import select
from SOM import *
import unittest

class Test(unittest.TestCase):

    def test_centroid(self):

        #self.assertRaises(centroid(-1, 2, (1,1)))
        #self.assertRaises(centroid(0, 0, (1,1)))
        c = centroid(1, 2, (-2,2))
        self.assertEqual(c.id, 1)
        self.assertEqual(len(c.loc), 2)
        self.assertTrue(2 >= c.loc[0] >= -2)
        self.assertTrue(2 >= c.loc[1] >= -2)


class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.som = SOM((5,5), 0.1)
        cls.som_flat = SOM((10,), 0.1)

    def test_init(cls):
        cls.assertEqual(len(cls.som.cen), 25)
        cls.assertEqual(cls.som.learning_rate, 0.1)
        cls.assertEqual(len(cls.som_flat.cen), 10)
        cls.assertEqual(cls.som_flat.learning_rate, 0.1)
        

if __name__ == '__main__':
    unittest.main()