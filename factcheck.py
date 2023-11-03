# factcheck.py

import re
import string
import torch
import numpy as np
import spacy
import gc

import nltk
from typing import List
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
from nltk import pos_tag

from transformers import AutoConfig

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

spacy_nlp = spacy.load("en_core_web_sm")

class FactExample:
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
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
        config = AutoConfig.from_pretrained(self.model.config.name_or_path)
        self.label_mapping = config.id2label

    # get_label_from_id - function to get the string label from the predicted class ID
    def get_label_from_id(self, class_id):
        labels = self.label_mapping
        return labels[class_id]        

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            max_token_length = 512
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True, max_length=max_token_length)
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits
        gc.collect()
  
        return probs

class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(object):

    def word_overlap(self, tokenized_facts: str, tokenized_passage: str):
        numerator = len(set(tokenized_facts) & set(tokenized_passage))
        denominator = min(len(set(tokenized_facts)), len(set(tokenized_passage)))
        overlap_coefficient = numerator / denominator
        return overlap_coefficient
    
    def evaluate_similarity(self, fact, passages) -> float:
        tokenized_facts = nltk.word_tokenize(fact)

        stem: bool = False
        remove_punctuation : bool = True
        remove_stop_words: bool = False
        similarity_metric: str = 'overlap'
        
        stemmer = nltk.stem.PorterStemmer()

        if remove_punctuation:
            tokenized_facts = [word for word in tokenized_facts if word not in string.punctuation]

        if stem:
            tokenized_facts = [stemmer.stem(word) for word in tokenized_facts]
        
        stop_words = set(nltk.corpus.stopwords.words('english'))

        if remove_stop_words:
            tokenized_facts = [word for word in tokenized_facts if word.lower() not in stop_words]

        # remove empty strings
        tokenized_facts = [word for word in tokenized_facts if word]

        results = []
        for passage in passages:
            passage_text = passage['title'] + ' ' + passage['text']
            tokenized_passage = nltk.word_tokenize(passage_text)

            tokenized_passage = [word for word in tokenized_passage if word]

            if remove_punctuation:
                tokenized_passage = [word for word in tokenized_passage if word not in string.punctuation]

            if stem:
                tokenized_passage = [stemmer.stem(word) for word in tokenized_passage]
        
            if remove_stop_words:
                tokenized_passage = [word for word in tokenized_passage if word.lower() not in stop_words]
        
            # similarity calculation
            if similarity_metric == 'overlap':
                sim = self.word_overlap(tokenized_facts, tokenized_passage)
            else:
                raise ValueError(f'Unsupported similarity metric: {similarity_metric}')
        
            # add similarity result to results list
            results.append(sim)

        # compute average similarity score
        average_similarity = np.average(results) if results else 0
        return average_similarity

    def predict(self, fact: str, passages: List[dict]) -> str:
        threshold: float = 0.60

        # compute average similarity score
        word_overlap_similarity = self.evaluate_similarity(fact, passages)
        
        # threshold Checking
        meets_threshold = word_overlap_similarity >= threshold
        prediction = ""
        if meets_threshold:
            prediction = "S"
        else:
            prediction = "NS"

        return prediction
class EntailmentFactChecker(object):
    def __init__(self, ent_model):
        self.ent_model : EntailmentModel = ent_model
        self.word_recall_fact_checker = WordRecallThresholdFactChecker()
        self.sent_tokenizer = PunktSentenceTokenizer()

    def clean_text(self, text):
        #text = text.encode("ascii", "ignore").decode()     # remove non-ASCII characters
        text = re.sub(r'<s>|</s>', '', text)  # remove <s> and </s> tokens
        # text = re.sub(r'""', '"', text)  # normalize double quotes
        # text = re.sub(r'"\s*\.|\."\s*"', '.', text)  # handle quotes around periods
        # text = re.sub(r'"\s*,|\,"\s*"', ',', text)  # handle quotes around commas
        # text = re.sub(r'\s+"|\s+"', ' ', text)  # handle quotes preceded/followed by whitespace
        # text = text.replace('"', '')  # remove any remaining double quotes
        # text = re.sub(r'\s+', ' ', text).strip()  # replace multiple spaces with a single space and strip leading/trailing spaces
        return text

    # is_coherent checks if a sentence is coherent or not so we can filter out the incoherent ones. 
    def is_coherent(self, sentence):
        # length based filtering
        if len(word_tokenize(sentence)) < 10:  # assuming minimum of 5 words for coherence
            return False
        
        # pos tagging
        tags = [tag for word, tag in pos_tag(word_tokenize(sentence))]
        essential_tags = ['NN', 'VB']  # nouns and Verbs
        if not any(tag in tags for tag in essential_tags):
            return False
        
        return True

    def chunk_text(self, text, max_length):
        # Clean the text first
        text = self.clean_text(text)
        
        # Use the tokenizer to encode the text, handling padding and truncation
        encoded_input = self.ent_model.tokenizer(
            text,
            #add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            add_special_tokens=False,
            max_length=max_length,   # Specify the max length for each chunk
            padding='max_length',    # Pad to max_length
            truncation=True,         # Truncate to max_length
            return_tensors='pt'      # Return PyTorch tensors
        )
        
        # Convert token IDs back to tokens to get a list of strings
        tokens = self.ent_model.tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0], skip_special_tokens=True)
        
        # Convert tokens to a single string, if necessary
        text_chunk = self.ent_model.tokenizer.convert_tokens_to_string(tokens)
        
        # # Convert token IDs back to tokens to get a list of strings
        # tokens = self.ent_model.tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0], skip_special_tokens=True)
        
        # # Convert tokens to a single string
        # text_chunk = self.ent_model.tokenizer.convert_tokens_to_string(tokens)
        
        # # TODO - find out if this helps or if I even need it?
        # text_chunk = text_chunk.encode('ascii', 'ignore').decode('ascii')

        # text_chunk = text_chunk.replace('\\"', '"')
        return text_chunk
        #return text_chunk.strip()
        # text = self.clean_text(text)
        # sentences = self.sent_tokenizer.tokenize(text)
        # return sentences

    def check_fact(self, fact, passages, threshold=0.25, max_length=512):
        max_entailment_score = 0.0
        most_entailing_sentence = ""

        max_contradiction_score = 0.0
        most_contradicting_sentence = ""

        passage_results = []
        word_overlap_decision = self.word_recall_fact_checker.predict(fact, passages)

        clean_total_full_passage_text = ""
        for passage in passages:
            full_text = passage['title'] + " is about " + passage['text']
            #full_text = passage['text']

            sentences = self.chunk_text(full_text, max_length)
            #sentences = [sentence for sentence in sentences if self.is_coherent(sentence)]
            #for sent in sentences:
            clean_sentences = self.clean_text(sentences)
            clean_total_full_passage_text += clean_sentences
            entailment_prob, neutral_prob, contradiction_prob = self.ent_model.check_entailment(fact, clean_sentences)
            # if neutral_prob > 0.50 and neutral_prob > entailment_prob and neutral_prob > contradiction_prob:
            #     return {
            #         "decision": word_overlap_decision,
            #         "max_entailment_score": max_entailment_score,
            #         "most_entailing_sentence": most_entailing_sentence,
            #         "most_contradicting_sentence": most_contradicting_sentence,
            #         "clean_total_full_passage_text": clean_total_full_passage_text,
            #         "og_passages": passages
            #     }
            
            if entailment_prob > neutral_prob or contradiction_prob > neutral_prob:
                if entailment_prob > max_entailment_score:
                    max_entailment_score = entailment_prob
                    most_entailing_sentence = full_text
                
                if contradiction_prob > max_contradiction_score:
                    max_contradiction_score = contradiction_prob
                    most_contradicting_sentence = full_text

            passage_decision = "S" if max_entailment_score > threshold else "NS"
            passage_results.append(passage_decision)

        decision_a = "S" if max_entailment_score > max_contradiction_score else word_overlap_decision
        decision_b = "S" if passage_results.count("S") > passage_results.count("NS") else word_overlap_decision
        #decision_c = "S" if passage_results.count("S") > 0 else "NS"
        decisions = [decision_a , decision_b] # [decision_b, decision_c, decision_e]

        decision = "S" if decisions.count("S") > decisions.count("NS") else word_overlap_decision

        return {
            "decision": decision,
            "max_entailment_score": max_entailment_score,
            "most_entailing_sentence": most_entailing_sentence,
            "max_contradiction_score": max_contradiction_score,
            "most_contradicting_sentence": most_contradicting_sentence,
            "clean_total_full_passage_text": clean_total_full_passage_text,
            "og_passages": passages
        }

    def check_fact_whole_passage(self, fact, passages, threshold=0.5, max_length=512, overlap=50):
        max_entailment_score = 0.0
        max_contradiction_score = 0.0
        ENTAILMENT_INDEX = 0
        CONTRADICTION_INDEX = 2
        most_entailing_sentence = ""
        most_contradicting_sentence = ""
        passage_results = []

        clean_total_full_passage_text = ""

        for passage in passages:
            full_passage_text = passage['title'] + " " + passage['text']

            clean_full_passage_text = self.clean_text(full_passage_text)
            clean_total_full_passage_text += clean_full_passage_text + "\n"
            
            entailment_probs, neutral_probs, contradiction_probs = self.ent_model.check_entailment(fact,clean_full_passage_text)

            if entailment_probs > max_entailment_score:
                max_entailment_score = entailment_probs
                most_entailing_sentence = full_passage_text

            passage_decision = "S" if max_entailment_score > max_contradiction_score else "NS"
            passage_results.append(passage_decision)

        decision_b = "S" if passage_results.count("S") > passage_results.count("NS") else "NS"
        decision_c = "S" if passage_results.count("S") > 0 else "NS"

        decisions = [decision_b, decision_c]
        decision = "S" if decisions.count("S") > decisions.count("NS") else "NS"

        return {
            "decision": decision,
            "max_entailment_score": max_entailment_score,
            "max_contradiction_score": max_contradiction_score,
            "most_entailing_sentence": most_entailing_sentence,
            "most_contradicting_sentence": most_contradicting_sentence,
            "clean_total_full_passage_text": clean_total_full_passage_text
        }
    
    def predict(self, fact: str, passages: List[dict], overlap_threshold=0.45, positive_threshold=0.25) -> str:
        max_length = 512
        word_overlap_similarity = self.word_recall_fact_checker.evaluate_similarity(fact, passages)
        if word_overlap_similarity < overlap_threshold:
            return "NS"
        
        cleaned_fact = self.clean_text(fact)
        result_c = self.check_fact(cleaned_fact, passages, threshold=positive_threshold, max_length=max_length)
        decision_c = result_c["decision"]

        return decision_c

# OPTIONAL
class DependencyRecallThresholdFactChecker(object):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations

