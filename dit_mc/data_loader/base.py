from abc import ABC

class DataLoader(ABC):
    def __init__(self, data_cfg, num_atoms, num_atoms_mean, max_num_graphs=None, debug=False):
        if max_num_graphs is None:
            print("Warning: max_num_graphs is set to default value of 1. This is likely unintended.")
            max_num_graphs = 1
            
        self.data_cfg = data_cfg
        self.num_atoms = num_atoms
        self.num_atoms_mean = num_atoms_mean
        self.max_num_graphs = max_num_graphs
        self.debug = debug

    def download(self):
        pass

    def next_epoch(self, split):
        raise NotImplementedError()
    
    def get_len(self, split):
        raise NotImplementedError()

    def get_sample(self):
        raise NotImplementedError()

    def shutdown(self):
        raise NotImplementedError()

    # Legacy way to access the dataset
    def __call__(self, split): 
        raise NotImplementedError()
