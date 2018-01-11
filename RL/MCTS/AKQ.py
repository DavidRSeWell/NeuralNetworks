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

#from RL.MCTS.Model import MCTS
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

    def __init__(self,name,info_tree,starting_stack):

        self.name = name

        self.info_tree = info_tree

        self.out_of_tree = False

        self.current_hand = None

        self.starting_stack = starting_stack

class AKQGameState(object):
    '''
    class used for simulating a simple AKQ poker game
    The game state needs to deal random cards to each player
    and to track the button
    '''

    def __init__(self,game_tree):

        self.player1 = None

        self.player2 = None

        self.deck = [3,2,1]

        self.game_tree = game_tree # the full game tree for for the information trees to reference

        self.init_game()

    def deal_hand(self):

        return random.choice(self.deck)

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

        if s.player == "chance" or s.parent == None:
            # the node is a change node
            pass



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

    def get_child_info(self,u_i,a):

        for child in u_i.children:
            if u_i.action == a:
                return child

        # should not reach this location

        raise Exception("g_child_info parent does not have child with action: " + str(a))

    def init_game(self):

        p1_tree = Tree() # info tree

        chance_node = AKQNode(player="chance",pot=1,cip=0)

        p1_tree.set_root(chance_node)

        p2_tree = Tree() # info tree

        p2_tree.set_root(self.game_tree.get_root())

        self.player1 = AKQPlayer(name="p1",info_tree=p1_tree,starting_stack=1)

        self.player2 = AKQPlayer(name="p2",info_tree=p2_tree,starting_stack=1)

    def reward(self,s):

        '''

        Takes in a leaf node and returns the reward to each player
        :param s:
        :return:

        '''

        r = {"p1":0,"p2":0}

        current_player = self.player1 if s.player == "p1" else self.player2

        if s.action == "fold":
            # the parent folded so the current player gets the pot
            r[s.parent.player] = s.parent.cip

            r[s.player] = s.pot + (current_player - s.cip)


        elif s.action == "check":

            # evaluate winner
            if (self.player1.current_hand > self.player2.current_hand):
                # p1 wins
                r["p1"] = s.pot + (self.player1.starting_stack - s.cip)

                r["p2"] = self.player2.starting_stack - s.cip

            else:

                r["p2"] = s.pot + (self.player2.starting_stack - s.cip)

                r["p1"] = self.player1.starting_stack - s.cip


        elif s.action == "call": # same as check?

            # evaluate winner
            if (self.player1.current_hand > self.player2.current_hand):
                # p1 wins
                r["p1"] = s.pot + (self.player1.starting_stack - s.cip)

                r["p2"] = self.player2.starting_stack - s.cip

            else:

                r["p2"] = s.pot + (self.player2.starting_stack - s.cip)

                r["p1"] = self.player1.starting_stack - s.cip

    def rollout(self,s):

        '''
            takes in a state
            gets action based off of a rollout policy - i.e random actions, ect...
            new state s' from G(s,a) - transition simulator
            return simulate(s')
        '''

        new_state = self.rollout_policy(s)

        return self.simulate(new_state) # recursive call

    def rollout_policy(self,s):

        '''

        Get a new action for the player based off the rollout policy
        :param s:
        :return:

        '''

        # for now just going to use a random rollout policy

        #possible_actions = [child.action for child in s.children]

        # just return the child node

        return random.choice(s.children)

    def select_uct(self,u_i):

        '''
            select action that maximizes
            Q(u,a) + c sqrt( log(N(u))/N(u,a) )

        '''

        N_U = u_i.visit_count

        possible_children = u_i.children

        current_max_child = None

        current_max = -1

        for child in possible_children:

            score = child.current_ev_value + 0.2*np.sqrt(np.log(N_U)/child.visit_count)

            if score > current_max:

                current_max = score

                current_max_child = child

        return current_max_child.action



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


        current_player = self.player1 if s.player == "p1" else self.player2

        if current_player.out_of_tree == True:

            return self.rollout(s)


        #infostate = self.get_info_state(s)

        NewInfoNode = InfoNode(current_player.current_hand,pot=s.pot,action=s.action,parent=s.parent,
                               cip=s.cip)

        action = None

        if InfoNode not in current_player.info_tree.get_nodes():

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

        u_i.visit_count += 1

        child_node = self.get_child_info(u_i,a)

        child_node.visit_count += 1

        child_node.current_ev_value += (r - child_node.current_ev_value)/child_node.visit_count

    def run(self,num_iterations):

        for i in range(num_iterations):

            self.deck = [3,2,1] # reshuffle the cards yo

            # deals cards to each player

            sb_card = self.deal_hand()

            self.player1.current_hand = sb_card

            self.deck.remove(sb_card)

            bb_card = self.deal_hand()

            self.player2.current_hand = bb_card

            s0 = self.game_tree.get_root()

            self.simulate(s0)













