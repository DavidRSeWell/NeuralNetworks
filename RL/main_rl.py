'''
module for running RL programs
'''

from RL.MCTS import Tree,Graph
import graphviz as gv

run_akq_game = 1
if run_akq_game:

    '''
        Define the game

    '''

    players = ["player1","player2"]

    AKQ_tree = Tree.Tree("player1",1)

    bet_node = AKQ_tree.add_node("player2","bet",1)
    check_node = AKQ_tree.add_node("player2","check",1)

    bet_node.add_node("player1","call",1)
    bet_node.add_node("player1","fold")

    bet_v_check = check_node.add_node("player1","bet",1)
    check_node.add_node("player1","check")

    bet_v_check.add_node("player2","call",1)
    bet_v_check.add_node("player2","fold")

    '''

        Define the graph

    '''

    graph_viz = gv.Digraph()

    graph = Graph.TreeGraph(AKQ_tree,graph_viz)

    full_graph = graph.graph_from_tree(AKQ_tree)

    print "ALLLL INNNNN"