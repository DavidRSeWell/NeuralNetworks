'''
    Tree class used for representing HU poker situations
'''

import graphviz as gv


class Tree:

    tree_index = 0

    def __init__(self):

        Tree.tree_index += 1

        self.my_tree_index = Tree.tree_index

        self.nodes = []

        self.node_index = 0

    def set_root(self,node):

        '''
        Init the tree with the first player in the list as the
        initial player for the root node
        :return:
        '''

        assert(self.node_index == 0) # this method should only be used for setting initial root

        self.nodes.insert(0,node)

    def add_node(self,node):

        '''
        Adds a node to the current tree
        :param player: string
        :param action: string
        :param amount: float
        :return: None
        '''

        self.node_index += 1

        node.node_index = self.node_index

        self.nodes.append(node) # add the new node to the list of nodes on the tree

        node.parent.children.append(node) # add the node to parents list of children

    def get_root(self):

        return self.nodes[0]

    def get_node(self,index):

        return self.nodes[index]

    def get_nodes(self):

        return self.nodes

    def get_tree_node(self,node):

        '''
            get node from outside of tree and returns node
            from current tree that has the same attributes
        '''

        for tree_node in self.nodes:

            if node.__dict__ == tree_node.__dict__:
                return tree_node

        return None

    def node_in_tree(self,node):

        '''
        checks to see if a node is already contained in the tree
        loop over each node in tree and checks if attributes are the same
        :param node:
        :return:
        '''

        for tree_node in self.nodes:

            if node.__dict__ == tree_node.__dict__:
                return True


        return False











