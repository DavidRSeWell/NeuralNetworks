'''
    Tree class used for representing HU poker situations
'''

import graphviz as gv


class Node:

    def __init__(self,player,pot):

        self.player = player

        self.pot = pot


class Tree:

    Nodeindex = 0 # class attribute used to track the index of nodes
    '''
        General Tree structure:
                A
                |
              / | |
            A1 A2 A3

        Each node is itself a Tree
        whith attributes player,pot
    '''

    def __init__(self,player,pot,parent=None,action=None,children=[]):

        self.player = player

        self.pot = pot

        self.action = action

        self.children = children

        self.parent = parent

        self.node_index = Tree.Nodeindex



    def set_root(self,player,init_pot):

        '''
        Init the tree with the first player in the list as the
        initial player for the root node
        :return:
        '''

        root = Node(player,init_pot)

        self.root = root


    def add_node(self,player,action,amount=None):

        '''
        Adds a node to the current tree
        :param player: string
        :param action: string
        :param amount: float
        :return: None
        '''

        new_pot = self.pot

        Tree.Nodeindex += 1

        if action == "bet" or action == "raise" or action == "call":

            try:
                assert amount > 0

                new_pot += amount

            except Exception,e:

                print "Error: Bet amount must be > 0"


        new_child = Tree(player,new_pot,parent=self,action=action)

        self.children.append(new_child)

        return new_child













