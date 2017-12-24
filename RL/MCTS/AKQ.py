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

from RL.MCTS.Model import MCTS


class AkqMC(MCTS):

    def __init__(self,tree):
        pass


    def is_leaf(self,node):

        return True if len(node.children) == 0 else False

    def rollout(self,node):


        if self.is_leaf(node):
            return node.current_value


        actions = self.avail_actions(node)

        random_action = random.choice(actions)

        roll_value = self.simulate(random_action,node)


    def avail_actions(self,node):

        pass

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

        '''
            Get next node based off of node and action s
            something like
            Game = new Game(tree=tree at current node)
            current_node = node
            while (current_node is not leaf)

                Game.take_random_action(node)

            return node.
        '''


    def main(self,num_iterations):
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
            if self.is_leaf(init_node):

                # is the value of the current node == 0?
                # if so then just rollout
                if (current_node.current_value == 0):

                    roll_value = self.rollout(current_node)

                    current_node.current_value = roll_value

                else:
                    # for each avail action from current node
                    # add a new state/node
                    for action in self.avail_actions(current_node):
                        self.tree.add_node(action)

                    current_node = current_node.children[0]

                    self.rollout(current_node)


    def equity(self,range1,range2):

        win_percent = range1["A"]*range2["K"] + range1["A"]*range2["Q"] +\
            range1["K"]*range2["Q"]

        tie_percent = range1["A"] * range2["A"] + range1["K"] * range2["K"] + \
                      range1["Q"] * range2["Q"]

