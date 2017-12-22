'''
Different MCTS implementations

'''

from abc import ABCMeta,abstractmethod

class MCTS(metaclass=ABCMeta):
    '''
        This is an abstract class for MCTS
    '''

    @abstractmethod
    def is_leaf(self,node):
        '''
        Take in node return True if node is leaf of tree
        :param node:
        :return:
        '''
        pass

    @abstractmethod
    def rollout(self,node):
        '''
        Peforms a simulation from the current tree node
        and estimates the value

        This needs to be overridden in the child class
        :param node:
        :return:
        '''
        pass

    @abstractmethod
    def avail_actions(self,node):

        pass

    @abstractmethod
    def simulate(self,action,node):
        pass

