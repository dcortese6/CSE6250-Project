import pickle


with open('data/model/model_data_x.pkl', 'rb') as f:
    x = pickle.load(f)
    
print(x)