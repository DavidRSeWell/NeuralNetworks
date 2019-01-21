'''
module for running RL programs
'''

from RL.MCTS import Game,Tree,Graph
from RL.MCTS.AKQ import  AKQGameState
from RL.MCTS.Node import Node
import graphviz as gv
import pandas as pd


run_simple_akq_game = 0
if run_simple_akq_game:
    tree = Tree.Tree()

    players = ["p1", "p2"]

    init_p1_cip = 0.0
    init_p2_cip = 0.0

    akq_game = Game.GameState(tree=tree, players=players, name='akq_game')

    akq_game.set_root(players[0], init_p1_cip, init_p2_cip)

    root = akq_game.tree.get_root()

    akq_game.new_action(current_index=0, player="p1", action={"check": 0})

    akq_game.new_action(current_index=1, player="p2", action={"bet": 1})
    akq_game.new_action(current_index=1, player="p2", action={"check": 0})

    akq_game.new_action(current_index=2, player="p1", action={"call": 1})
    akq_game.new_action(current_index=2, player="p1", action={"fold": 0})


    GameState = AKQGameState(tree)

    new_graph = gv.Digraph(format="png")

    AKQGraph = Graph.TreeGraph(tree=akq_game.tree, graph=new_graph)

    AKQGraph.create_graph_from_tree()

    AKQGraph.graph.render('data/img/simple_akq_game')

    p1_policy, p2_policy = GameState.run(100000000)

    p1_ev_matrix = []

    for node in GameState.player1.info_tree.nodes:

        if node.player == "chance":
            continue

        current_hand = node.player_hand

        policy = p1_policy[node.node_index]

        for action in policy.keys():
            ev = policy[action]['ev']

            p1_ev_matrix.append(
                ['player 1', 'node: ' + str(node.node_index), 'hand:' + str(current_hand), action, 'value: ' + str(ev)])

    ev_df_1 = pd.DataFrame(p1_ev_matrix)

    p2_ev_matrix = []

    for node in GameState.player2.info_tree.nodes:

        if node.player != "p2":
            continue

        current_hand = node.player_hand

        policy = p2_policy[node.node_index]

        for action in policy.keys():
            ev = policy[action]['ev']

            p2_ev_matrix.append(
                ['player 2', 'node: ' + str(node.node_index), 'hand:' + str(current_hand), action, 'value: ' + str(ev)])

    ev_df_2 = pd.DataFrame(p2_ev_matrix)

    ev_df_1.to_csv('/Users/befeltingu/NeuralNetworks/RL/data/simple_akq_1.csv')

    ev_df_2.to_csv('/Users/befeltingu/NeuralNetworks/RL/data/simple_akq_2.csv')

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

    init_p1_cip = 0.0
    init_p2_cip = 0.0

    akq_game = Game.GameState(tree=tree,players=players,name='akq_game')

    akq_game.set_root(players[0],init_p1_cip,init_p2_cip)

    root = akq_game.tree.get_root()

    akq_game.new_action(current_index=0,player="p1",action={"bet":1})
    akq_game.new_action(current_index=0,player="p1",action={"check":0})

    akq_game.new_action(current_index=1,player="p2",action={"call":1})
    akq_game.new_action(current_index=1,player="p2",action={"fold":0})

    akq_game.new_action(current_index=2,player="p2",action={"bet":1})
    akq_game.new_action(current_index=2,player="p2",action={"check":0})

    akq_game.new_action(current_index=5,player="p1",action={"call":1})
    akq_game.new_action(current_index=5,player="p1",action={"fold":0})

    GameState = AKQGameState(tree)

    new_graph = gv.Digraph(format="png")

    AKQGraph = Graph.TreeGraph(tree=akq_game.tree,graph=new_graph)

    AKQGraph.create_graph_from_tree()

    AKQGraph.graph.render('data/img/test')

    p1_policy, p2_policy = GameState.run(1000)


    p1_ev_matrix = []

    for node in GameState.player1.info_tree.nodes:

        if node.player == "chance":
            continue

        current_hand = node.player_hand

        policy = p1_policy[node.node_index]

        for action in policy.keys():

            ev = policy[action]['ev']

            p1_ev_matrix.append(['player 1', 'node: ' + str(node.node_index), 'hand:' + str(current_hand) , action , 'value: ' + str(ev)])

    ev_df_1 = pd.DataFrame(p1_ev_matrix)

    p2_ev_matrix = []

    for node in GameState.player2.info_tree.nodes:

        if node.player != "p2":
            continue

        current_hand = node.player_hand

        policy = p2_policy[node.node_index]

        for action in policy.keys():
            ev = policy[action]['ev']

            p2_ev_matrix.append(
                ['player 2', 'node: ' + str(node.node_index), 'hand:' + str(current_hand), action, 'value: ' + str(ev)])

    ev_df_2 = pd.DataFrame(p2_ev_matrix)

    ev_df_1.to_csv('/Users/befeltingu/NeuralNetworks/RL/data/ev_1.csv')

    ev_df_2.to_csv('/Users/befeltingu/NeuralNetworks/RL/data/ev_2.csv')

