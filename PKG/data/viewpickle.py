import pickle


with open('/home/ravindu.nagasinghe/GithubCodes/RaviPP/data/trained_graph.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)