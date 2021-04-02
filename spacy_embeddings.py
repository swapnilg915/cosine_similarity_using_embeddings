""" 
date: 2nd April 2021
aim: to calculate cosine similarity between 2 sentences using spacy
note: 
	used spacy version = 2.2.0 


"""


import os
import json
import re
import numpy as np 
import spacy
from spacy.lang.en import English
spacy_en = English(disable=['parser', 'ner'])
from spacy.lang.en.stop_words import STOP_WORDS as stopwords_en


class SpacySimilarity(object):

	def __init__(self):
		self.spacy_large_model = spacy.load("en_core_web_lg")	

	def clean_text(self, text):
		try:
			text = str(text)
			text = re.sub(r"[^A-Za-z0-9]", " ", text)
			text = re.sub(r"\s+", " ", text)
			text = text.lower().strip()
		except Exception as e:
			print("\n Error in clean_text --- ", e,"\n ", traceback.format_exc())
			print("\n Error sent --- ", text)
		return text

	def get_lemma_tokens(self, text):
		return " ".join([tok.lemma_.lower().strip() for tok in spacy_en(text) if (tok.lemma_ != '-PRON-' and tok.lemma_ not in stopwords_en and len(tok.lemma_)>1)])

	def cleaning_pipeline(self, text):
		text = self.clean_text(text)
		text = self.get_lemma_tokens(text)
		return text

	def cos_sim(self, vector_1, vector_2):
		return np.inner(vector_1, vector_2) / (np.linalg.norm(vector_1) * (np.linalg.norm(vector_2)))

	def main(self, sent1, sent2):
		sent1_cleaned = self.cleaning_pipeline(sent1)
		sent2_cleaned = self.cleaning_pipeline(sent2)
		sent1_vector = self.spacy_large_model(sent1_cleaned).vector
		sent2_vector = self.spacy_large_model(sent2_cleaned).vector
		cosine_sim = self.cos_sim(sent1_vector, sent2_vector)
		print("\n spacy cosine similarity = ", cosine_sim)

if __name__ == '__main__':
	obj = SpacySimilarity()
	sent1 = "booking a flight is very easy"
	sent2 = "readinga book is a good habit"
	obj.main(sent1, sent2)