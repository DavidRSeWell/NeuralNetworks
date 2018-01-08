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
from RL.MCTS.Node import AKQNode,InfoNode
from RL.MCTS.Tree import Tree



class ExtensiveFormMCTS(object):

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

class AKQPlayer(object):

    def __init__(self,name,info_tree):

        self.name = name

        self.info_tree = info_tree

        self.out_of_tree = False

        self.current_hand = None

class AKQGameState(object):

    '''
    class used for simulating a simple AKQ poker game
    The game state needs to deal random cards to each player
    and to track the button
    '''

    def __init__(self,game_tree):

        self.player1 = None

        self.player2 = None

        self.deck = ['A','K','Q']

        self.game_tree = game_tree # the full game tree for for the information trees to reference

    def deal_hand(self):

        return random.choice(self.deck)

    def expand_tree(self,current_player,info_state):

        '''
        Expands the info tree of the current player
        :param current_player:
        :param state:
        :return:
        '''
        pass

    def get_info_state(self, s):

        '''

        info state for the AKG game is (actions,hand)
        actions are all previous actions to this node

        For the AKQ I dont think we need to keep track of
        actions so the info stat is only going to have
        the additional hand value

        Append
        :param s:
        :return:

        '''



        pass

    def get_new_state(self,s,a):

        node_children = s.children

        for child in node_children:

            if child.action == a:

                return child

            else:
                continue

        # we should not reach this line of code
        # the function should always be able to return a new state

        raise Exception("get_new_state was not able to find a child with the given action")

    def init_game(self):

        p1_tree = Tree() # info tree

        chance_node = AKQNode(player="chance",pot=1)

        p1_tree.set_root(chance_node)

        p2_tree = Tree() # info tree

        p2_tree.set_root(self.game_tree.get_root())

        self.player1 = AKQPlayer(name="p1",info_tree=p1_tree)

        self.player2 = AKQPlayer(name="p2",info_tree=p2_tree)

    def reward(self,s):
        '''
        Takes in a leaf node and returns the reward to each player
        :param s:
        :return:
        '''

        pass

    def search(self,game):

        '''
            While within budget
                Sample initial game state
                simulate(s_o)
            end

            return policy
        '''
        pass

    def rollout(self,s):

        '''
            takes in a state
            gets action based off of a rollout policy - i.e random actions, ect...
            new state s' from G(s,a) - transition simulator
            return simulate(s')
        '''
        pass

    def rollout_policy(self,s):

        '''

        Get a new action for the player based off the rollout policy
        :param s:
        :return:

        '''
        pass

    def select_uct(self,u_i):
        '''
            select action that maximizes
            Q(u,a) + c sqrt( log(N(u))/N(u,a) )

        '''
        pass

    def simulate(self,s):
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

        if s.is_leaf == True:

            return self.reward(s)


        current_player = s.player

        if current_player.out_of_tree == True:

            return self.rollout(s)


        #infostate = self.get_info_state(s)

        NewInfoNode = InfoNode(s,current_player.current_hand)

        action = None

        if InfoNode not in current_player.info_tree.__nodes:

            current_player.info_tree.add_node(NewInfoNode)

            action = self.rollout_policy(s)


        else:

            action = self.select_uct(NewInfoNode)


        next_state = self.get_new_state(s,action)

        r = self.simulate(next_state)

        self.update(NewInfoNode,action,r)

        return r








        pass

    def update(self,u_i, a, r):
        '''
        N(u_i) += 1
        N(u,a) += 1
        Q(u,a) += (r - Q(u,a)) / N(u,a)
        '''
        pass

    def run(self,num_iterations):

        for i in range(num_iterations):

            self.deck = ['A','K','Q'] # reshuffle the cards yo

            # deals cards to each player

            sb_card = self.deal_hand()

            self.player1.current_hand = sb_card

            self.deck.remove(sb_card)

            bb_card = self.deal_hand()

            self.player2.current_hand = bb_card

            s0 = self.game_tree.get_root()

            self.simulate(s0)













