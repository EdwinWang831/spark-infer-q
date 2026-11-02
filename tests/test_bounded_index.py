import unittest
from spark_infer_q.bounded_index import validate_geometry


class Geometry(unittest.TestCase):
    def test_prepare_and_incomplete_block_must_survive(self):
        self.assertEqual(validate_geometry(2048),2048)
        self.assertEqual(validate_geometry(1028),1028)
        for rows in (1024,1027,2047,0,-4,True,2048.0):
            with self.subTest(rows=rows),self.assertRaises(ValueError):validate_geometry(rows)

    def test_invalid_prepare_and_ratio_are_rejected(self):
        for args in ((2048,0,4),(2048,1024,0),(2048,2048,4)):
            with self.subTest(args=args),self.assertRaises(ValueError):validate_geometry(*args)


if __name__=='__main__':unittest.main()
