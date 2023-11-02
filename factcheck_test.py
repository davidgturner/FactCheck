import unittest

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict
from factcheck import *

class TestFactChecker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize the model and tokenizer
        model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        ent_tokenizer = AutoTokenizer.from_pretrained(model_name)
        ent_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        cls.ent_model = EntailmentModel(ent_model, ent_tokenizer)
        cls.fact_checker = EntailmentFactChecker(ent_model)

    def test_fact_1(self):
        fact = "The Fischer Research Laboratory was founded in 1936."
        passages = [...]  # Load the passages for this fact
        prediction = self.fact_checker.predict(fact, passages)
        self.assertEqual(prediction, "NS")

    def test_fact_2(self):
        fact = "Maracaibo is in Venezuela."
        passages = [...]  # Load the passages for this fact
        prediction = self.fact_checker.predict(fact, passages)
        self.assertEqual(prediction, "S")

if __name__ == '__main__':
    unittest.main()