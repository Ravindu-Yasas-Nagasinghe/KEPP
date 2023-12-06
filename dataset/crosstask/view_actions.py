import numpy as np
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
data = np.load('/home/ravindu.nagasinghe/GithubCodes/PDPP/PDPP/dataset/crosstask/actions_one_hot.npy')
print(data)

np.load = np_load_old