import os, traceback
from time import time
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

class wmdistance(object):
    
    def __init__(self):
        self.stopwords = stopwords.words('english')
        self.w2v_path = '/home/swapnil/Downloads/GoogleNews-vectors-negative300.bin.gz'
        self.loadW2V()
        self.w2v_model.init_sims(replace=True)
        print("\nloaded word2vec model : ")
    
    
    def loadW2V(self):
        if not os.path.exists(self.w2v_path):
            raise ValueError("SKIP: You need to download the google news model")
        self.w2v_model = KeyedVectors.load_word2vec_format(self.w2v_path, limit=300000, binary=True)

    def createWmdDataset(self, sent):
        return [wrd for wrd in sent.split() if wrd not in self.stopwords]

    def getWmDistance(self, sent_1, sent_2):
        distance = ''
        try:
            sent_1 = self.createWmdDataset(sent_1)
            sent_2 = self.createWmdDataset(sent_2)
            distance = self.w2v_model.wmdistance(sent_1, sent_2)
            print("\n the word movers distance is => ",distance)
        except Exception as e:
            print("\n Error in getWmDistance --- ", e, "\n", traceback.format_exc())
        return distance
    
if __name__ == '__main__':
    obj = wmdistance()
    sent_1 = 'Obama speaks to the media in Illinois'
    sent_2 = 'The president greets the press in Chicago'
    obj.getWmDistance()