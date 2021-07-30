
import copy
from collections.abc import Mapping

class ParaSet(Mapping):
    """ 
    A parameter set used to store the setup for the layers and modules
    """
    def __init__(self,**kwargs):
        for key, value in kwargs.items():
            self.__setattr__( key, value)
    
    def clone(self, **kwargs):
        ins = copy.deepcopy(self)
        for key, value in kwargs.items():
            ins.__setattr__( key, value)
        return ins
    
    def update(self, dicts):
        for key, val in dicts.items():
            self.__setattr__( key, val)
        return self.__dict__
    
    def __iter__(self):
        for key in self.__dict__.keys():
            yield key
        
    def __len__(self):
        return len(self.__dict__)
    
    def __getitem__(self, item):
        return self.__dict__[item]
    
    def __call__(self, **kwargs):
        for key, value in kwargs.items():
    	    self.__setattr__( key, value)

#if __name__ == '__main__':
#
#	c = cfg.clone()
