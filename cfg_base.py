
import copy
from collections.abc import Mapping

class cfg_base(Mapping):
    def __init__(self, cfg_id= None, decorator = None, **kwargs):
        self.id = cfg_id
        self.decorator = decorator
        for key, value in kwargs.items():
            self.__setattr__( key, value)
    
    def clone(self, **kwargs):
        ins = copy.deepcopy(self)
        for key, value in kwargs.items():
            ins.__setattr__( key, value)
        return ins
    
    def __iter__(self):
        for key in self.__dict__.keys():
            yield key
    
    def __len__(self):
        return len(self.__dict__)
    
    def __getitem__(self, item):
        return self.__dict__[item]
    
    def decoration(self):
        return self.id, self.decorator(self)
    
    def __call__(self, **kwargs):
        for key, value in kwargs.items():
    	    self.__setattr__( key, value)

#if __name__ == '__main__':
#
#	c = cfg.clone()
