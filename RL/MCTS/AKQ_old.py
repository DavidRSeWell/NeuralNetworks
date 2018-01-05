

import random
import numpy as np

from RL.MCTS.Model import MCTS
from RL.MCTS.Node import Node



class AkqMC(MCTS):

    def __init__(self,tree):
        self.tree = tree

    def rollout_random(self,node,num_iterations=10):

        '''
        Perform rollout by taking random actions from the current
        node until you have reached a leaf node
        :param node:
        :param num_iterations: Number of times to run to completion
        :return: average value over the iterations. V/num_iteration
        '''

        tot_value = 0

        for i in range(num_iterations):

            current_node = node

            while(True):

                if current_node.is_leaf:

                    tot_value += self.node_value(current_node)

                    break

                actions = self.avail_actions(current_node)

                random_action = random.choice(actions)

                current_node = self.simulate(random_action,current_node)

        return tot_value / float(num_iterations)

    def avail_actions(self,node):
        '''
        The available actions depend directly on what the parent action was
        For the AKQ game the bet option is only bet 1
        IF BET:
            avail: [call,fold]

        ELIF CHECK:
            avail: [bet 1, check]
        :param node:
        :return:
        '''

        if node.action == "bet":

            return ["call","fold"]

        elif node.action == "check":

            return ["bet","check"]

    def node_value(self,node):

        '''

        Assumes the node is a leaf node
        Value depends on the action to get there

        if parent action was fold then the player for the
        leaf node will have a value of S + P
        and the other player will have a value of S

        if parent action is a call then the value for both players
        will be their Stack plus their share of the pot

        For evaluating a call we only care about the current ranges of
        the players

        :param node:

        :return:

        '''

    def get_opponent(self,player):

        if player == "player1":

            return "player2"

        else:

            return "player1"

    def simulate(self,action,node):

        '''
        For the AKQ game we are just going to implement a simple
        random rollout. That means we just choose a random action
        for each player until we are at a leaf node and then return
        the value
        :param action:
        :param node:
        :return:
        '''

        current_player = self.get_opponent(node.player)

        current_pot = node.pot

        is_leaf = False

        range2 = node.range1 # this will change dependent on the action

        if action == "bet":

            current_pot += 1

            range2 = self.get_random_range(range2)

        if action == "check" and node.action == "check":

            is_leaf = True # It was checked down

        elif action == "call":

            current_pot += 1

            range2 = self.get_random_range(range2)

            is_leaf = True

        elif action == "fold":

            is_leaf = True

        new_node = Node(current_player,current_pot,parent=node,action=action,
                        is_leaf=is_leaf,range1=node.range2,range2=range2)

        return new_node

    def equity(self,range1,range2):

        win_percent = range1["A"]*range2["K"] + range1["A"]*range2["Q"] +\
            range1["K"]*range2["Q"]

        tie_percent = range1["A"] * range2["A"] + range1["K"] * range2["K"] + \
                      range1["Q"] * range2["Q"]

        return (win_percent,tie_percent)

    def get_random_range(self,range):

        '''

        Takes the current range and returns a new range
        with randomly selected % of each card in that range
        :param range:
        :return:

        '''

        new_range = {"A":np.random.random(),"K":np.random.random(),"Q":np.random.random()}

        return new_range

    def update_pot(self,action,pot):

        pass


    def main(self, num_iterations):
        '''
                main algorithm loop
                4 main parts
                1 - Tree traversal
                2 - node expansion
                3 - rollout
                4 - backpropagation
                :return:
                '''

        init_node = None

        current_node = init_node

        for i in range(num_iterations):

            current_node.visit_count += 1

            # check is current node a leaf
            if init_node.is_leaf:

                # is the value of the current node == 0?
                # if so then just rollout
                if (current_node.current_value == 0):

                    roll_value = self.rollout(current_node)

                    current_node.current_value += roll_value

                else:
                    # for each child choose one with largest UCB1
                    # add a new state/node

                    for action in self.avail_actions(current_node):

                        new_player = self.get_opponent(current_node.player)



                        new_node= Node()

                        self.tree.add_node(action)

                    current_node = current_node.children[0]

                    self.rollout(current_node)

