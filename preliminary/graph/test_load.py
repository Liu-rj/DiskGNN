from dgl.data.utils import load_graphs
import time
import os
from tqdm import tqdm


path = '/data/offline_subgraph/'
files = os.listdir(path)
start = time.time()
for file in tqdm(files):
    glist, _ = load_graphs(os.path.join(path, file))
end = time.time()
print(end - start)