'''
    Node class module used as part of a tree data structure

'''



class Node(object):

    '''

        General Tree structure:
                A
                |
              / | |
            A1 A2 A3

        Each node is itself a Tree
        whith attributes player,pot

    '''

    def __init__(self,player,pot,parent=None,action=None,node_index=None,
                 is_leaf=False,range1={},range2={}):

        self.node_index = node_index

        self.player = player

        self.pot = pot

        self.action = action

        self.children = []

        self.parent = parent

        self.visit_count = 0 # number of times the node has been visited in MCTS

        self.current_ev_value = 0 # total value of node for current player

        #self.current_ucb1 = 0 # average ev + 2 * sqrt ( ln (total iterations) / visit_count)

        self.range1 = range1 # the range of the current player

        self.range2 = range2 # the range of the parent

        self.is_leaf = is_leaf


class InfoNode(object):

    '''

    Nodes used in incomplete information games

    '''

    def __init__(self, player, pot, parent=None, action=None, node_index=None,
                 is_leaf=False):

        self.node_index = node_index

        self.player = player

        self.pot = pot

        self.action = action

        self.children = []

        self.parent = parent

        self.visit_count = 0  # number of times the node has been visited in MCTS

        self.current_ev_value = 0  # total value of node for current player

        # self.current_ucb1 = 0 # average ev + 2 * sqrt ( ln (total iterations) / visit_count)

        self.is_leaf = is_leaf


class AKQNode(object):
    '''

            General Tree structure:
                    A
                    |
                  / | |
                A1 A2 A3

            Each node is itself a Tree
            whith attributes player,pot

        '''

    def __init__(self, player, pot, parent=None, action=None, node_index=None,
                 is_leaf=False):

        self.node_index = node_index

        self.player = player

        self.pot = pot

        self.action = action

        self.children = []

        self.parent = parent

        self.visit_count = 0  # number of times the node has been visited in MCTS

        self.current_ev_value = 0  # total value of node for current player, not used in extensive form games

        # self.current_ucb1 = 0 # average ev + 2 * sqrt ( ln (total iterations) / visit_count)

        self.is_leaf = is_leaf