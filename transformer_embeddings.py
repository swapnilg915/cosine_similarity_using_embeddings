"""
in case of sentence embeddings we dont need to clean the text, as sentence transformer models returns the contexual embeddings based on the meaning of the sentence, by removing stopwords/punctuations it disturbs the original meaning of the text
"""

import numpy as np
from sentence_transformers import SentenceTransformer

#lets load the pre-trained transformer model
# here we are using "" model. we can use any of the 124 pre-trained models available on huggingface model hub: https://huggingface.co/sentence-transformers
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

jd = 'machine learning engineer'
resume = 'lead data scientist'

#get the embeddings for the sentences
#Sentences are encoded by calling model.encode()
jd_embeddings = model.encode(jd)
resume_embeddings = model.encode(resume)

def cos_sim(a, b):
	return np.inner(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))

sim = cos_sim(jd_embeddings, resume_embeddings)
print("\n cosine simlarity : ", sim)
