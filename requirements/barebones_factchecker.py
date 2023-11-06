import torch
from typing import List
import numpy as np
import spacy
import gc

class FactExample:

    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            outputs = self.model(**inputs)
            logits = outputs.logits

        raise Exception("Not implemented")

        del inputs, outputs, logits
        gc.collect()

        # TODO return something

class FactChecker(object):

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Don't call me, call my subclasses")

class EntailmentFactChecker(object):
    def __init__(self, ent_model):
        self.ent_model = ent_model
		# TODO implement

    def predict(self, fact: str, passages: List[dict]) -> str:
		# TODO implement
        raise Exception("Implement me")



