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

                Strategy - The current strategy implemented on the current tre






'''


class Player:

    def __init__(self,chips):

        self.chips = chips


class Tree:

    def __init__(self,struct):

        self.struct = struct


class Node:

    def __init__(self):

        pass