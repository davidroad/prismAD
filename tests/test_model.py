import unittest
from prism_ad.demo import run
class NativeModelTests(unittest.TestCase):
    def test_synthetic_units_ledger_gradients_and_genetic_control(self):
        result=run(with_model=True)
        self.assertEqual(result["model_check"],"passed")
if __name__=="__main__":unittest.main()
