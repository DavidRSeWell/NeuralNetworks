'''
    Tree class used for representing HU poker situations
'''

import graphviz as gv


class Tree(object):

    NodeIndex = 0 # name this node count?

    def __init__(self,nodes=[]):

        self.__nodes = nodes

    def set_root(self,node):

        '''
        Init the tree with the first player in the list as the
        initial player for the root node
        :return:
        '''

        assert(Tree.NodeIndex == 0) # this method should only be used for setting initial root

        self.__nodes.insert(0,node)

    def add_node(self,node):

        '''
        Adds a node to the current tree
        :param player: string
        :param action: string
        :param amount: float
        :return: None
        '''

        Tree.NodeIndex += 1

        node.node_index = Tree.NodeIndex

        self.__nodes.append(node) # add the new node to the list of nodes on the tree

        node.parent.children.append(node) # add the node to parents list of children

    def get_root(self):

        return self.__nodes[0]

    def get_node(self,index):

        return self.__nodes[index]

    def get_nodes(self):

        return self.__nodes









