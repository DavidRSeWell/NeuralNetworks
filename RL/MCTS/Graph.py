'''
    Module containing the Graph class
'''

import graphviz as gv


class TreeGraph:

    '''
        Class used for providing a graphical representation of graphs
    '''
    def __init__(self,tree,graph=None):

        self.tree = tree

        self.graph = graph



    def graph_from_tree(self,tree):

        '''
                Takes the current tree representation of a tree and
                converts it to a graphiz graph

                :return:
        '''

        # first add root node

        if len(tree.children) == 0:

            label = tree.player + ' \\n ' + 'Pot: ' + str(tree.pot)

            self.graph.node(str(tree.node_index),label)

            self.graph.edge(str(tree.node_index),str(tree.parent.node_index),str(tree.action))

        else:

            for child in tree.children:

                label = tree.player + ' \\n ' + 'Pot: ' + str(tree.pot)

                self.graph.node(str(child.node_index),label)

                self.graph.edge(str(child.node_index), str(child.parent.node_index), str(child.action))

                self.graph_from_tree(child)

    def create_graph_from_tree(self,tree):

        '''

            Takes the current tree representation of a tree and
            converts it to a graphiz graph

        :return:
        '''

        # first add root node
        self.graph.node('0','p1 \\n Pot: ' + str(self.tree.pot))


        for child in self.tree.children:

            label = child.player + ' \\n ' + child.action

            self.graph.node(str(child.node_index),label)

            self.graph.edge(str(child.node_index))


    def add_nodes(self,graph, nodes):

        for n in nodes:

            if isinstance(n, tuple):
                graph.node(n[0], **n[1])
            else:
                graph.node(n)

        return graph


    def add_edges(self,graph, edges):

        for e in edges:
            if isinstance(e[0], tuple):
                graph.edge(*e[0], **e[1])
            else:
                graph.edge(*e)

        return graph
