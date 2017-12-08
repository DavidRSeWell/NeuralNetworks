'''
    Tree class used for representing HU poker situations
'''


class Node:

    def __init__(self,player,pot):
        self.player = player
        self.pot = pot


class Tree:

    def __init__(self,players):
        self.players = players
        self.init()

    def init(self):

        '''
        Init the tree with the first player in the list as the initial player for the root node
        :return:
        '''
        return None