'''
    MCTS solution to the AKQ game
'''

'''
    Software design:

        Player:

            Props:

                Range: {A,Q,K}

                Chips: number of big blinds

        Tree:

            Props:

                Root:

                Struct: - The current structure of the tree. { a: {b,c}, b: {c,d} ...

            Actions:

                addNode

                getNode

                getChildren

        Nodes:

            Properties:

                Pot: Size of the current pot

                Player: Player whos action it is

                Actions: Possible actions



        Solver:

            Props:

                Type: - Type of solver it is. MCTS ect...

                Tree: - The current tree

                Strategy - The current strategy implemented on the current tree

'''

import random
import numpy as np

from RL.MCTS.Model import MCTS
from RL.MCTS.Node import AKQNode
from RL.MCTS.Tree import Tree



class ExtensiveFormMCTS():

    def __init__(self):

        pass

    def search(game):
        '''
            While within budget
                Sample initial game state
                simulate(s_o)
            end

            return policy
        '''
        pass

    def rollout(s):
        '''
            takes in a state
            gets action based off of a rollout policy - i.e random actions, ect...
            new state s' from G(s,a) - transition simulator
            return simulate(s')
        '''
        pass

    def simulate(s):
        '''
            Takes in a state

            if state.terminal == True:
                return reward

            Player = player(s)
            if Player.out_of_tree == True:
                return rollout(s)
            InfoState = information_function(s) maps state to info state
            if InfoState not in PlayerTree:
                Expand(PlayerTree,InfoState)
                a = rollout_policy
                Player.out_of_tree = True
            else:
                a = select(InfoState)
            s' = G(s,a)
            r = simulate(s')
            update(InfoState,a,r)
            return r
        '''


        pass

    def select_uct(u_i):
        '''
            select action that maximizes
            Q(u,a) + c sqrt( log(N(u))/N(u,a) )

        '''
        pass

    def update(u_i, a, r):
        '''
        N(u_i) += 1
        N(u,a) += 1
        Q(u,a) += (r - Q(u,a)) / N(u,a)
        '''
        pass



class AKQGameState(object):

    '''
    class used for simulating a simple AKQ poker game
    The game state needs to deal random cards to each player
    and to track the button
    '''

    def __init__(self,players):

        self.players = players

        self.deck = ['A','K','Q']


    def deal_hand(self):

        return random.choice(self.deck)


    def run(self,num_iterations):

        sb_player = random.choice(self.players)

        if self.players[0] == sb_player:

            bb_player = self.players[1]

        else:

            bb_player = self.players[0]

        for i in range(num_iterations):

            self.deck = ['A','K','Q'] # reshuffle the cards yo

            # deals cards to each player

            sb_card = self.deal_hand()

            self.deck.remove(sb_player)

            bb_card = self.deal_hand()








