'''
module for running RL programs
'''

from RL.MCTS import Game,Tree,Graph
from RL.MCTS.AKQ import  AKQGameState
from RL.MCTS.Node import Node
import graphviz as gv


run_akq_game = 1
if run_akq_game:

    '''
        Define the game

    '''
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

        Define the graph

    graph_viz = gv.Digraph()

    graph = Graph.TreeGraph(AKQ_tree,graph_viz)

    full_graph = graph.graph_from_tree(AKQ_tree)

    '''

    tree = Tree.Tree()

    players = ["p1","p2"]

    init_p1_cip = 0.5
    init_p2_cip = 0.5

    akq_game = Game.GameState(tree=tree,players=players,name='akq_game')

    akq_game.set_root(players[0],init_p1_cip,init_p2_cip)

    root = akq_game.tree.get_root()

    akq_game.new_action(current_index=0,player="p1",action={"bet":1})
    akq_game.new_action(current_index=0,player="p1",action={"check":0})

    akq_game.new_action(current_index=1,player="p2",action={"call":1})
    akq_game.new_action(current_index=1,player="p2",action={"fold":0})

    akq_game.new_action(current_index=2,player="p2",action={"bet":1})
    akq_game.new_action(current_index=2,player="p2",action={"check":0})

    akq_game.new_action(current_index=5,player="p2",action={"call":1})
    akq_game.new_action(current_index=5,player="p2",action={"fold":0})


    GameState = AKQGameState(tree)



    p1_policy , p2_policy = GameState.run(100)

    #new_graph = gv.Digraph(format="png")


    #AKQGraph = Graph.TreeGraph(tree=akq_game.tree,graph=new_graph)

    #label = tree.player + ' \\n ' + 'Pot: ' + str(tree.pot)

    #self.graph.node(str(tree.node_index), label)

    #AKQGraph.graph.node

    #AKQGraph.graph_from_tree(akq_game.tree.get_root())

    #AKQGraph.create_graph_from_tree()

    #AKQGraph.graph.render('data/img/test')

    print "Game set match "