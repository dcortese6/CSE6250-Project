import gensim
from gensim.models import Word2Vec

class Model():
    def __init__(self):
        # self.data = data
        # self.vector_size = vector_size
        # self.window = window
        pass
        
    def get_embeddings(self, data, vector_size=100, window=5):
        model = Word2Vec(data, 
                        min_count = 1, 
                        vector_size = vector_size, 
                        window = window)
        
        return model
    
    