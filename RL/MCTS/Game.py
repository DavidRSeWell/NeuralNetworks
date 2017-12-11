'''

    Class for maintaing the state of the game and also for
    containing most things external to the special instance
    of the game itself
'''

from RL.MCTS import Tree,Node


class GameState(object):

    '''
        A game takes in actions from the environment
        and maintains the tree,nodes and graph based off these
    '''

    def __init__(self,tree,players,graph=None,name='',solver=None):

        self.name = name

        self.players = players

        self.tree = tree

        self.graph = graph

        self.solver = solver

        self.assert_init()

    def assert_init(self):

        assert (len(self.players) > 0)

        print "Finished initializing the game state"


    def set_root(self,player,pot):

        new_node = Node.Node(player,pot=pot,node_index=0)

        self.tree.set_root(new_node)


    def new_action(self,current_index,player,action):

        '''
        Method that takes in the intended actiont to take
        and adds it to the game state.

        :param player: string
        :param action: dict : {type:'',amount: float}
        :return:
        '''

        current_node = self.tree.get_node(current_index)

        new_pot = self.get_new_pot(current_node.pot,action)

        opponent = self.get_opponent(player)

        new_node = Node.Node(opponent,pot=new_pot,parent=current_node,action=action)

        self.tree.add_node(new_node)


    def get_new_pot(self,pot,action):

        '''
        Return new pot size based off a new action
        :param pot:
        :param action:
        :return:
        '''

        action_type = action.keys()[0]

        amount = action.values()[0]

        new_pot = pot

        if action_type in  ("bet","raise","call"):

            try:
                assert amount > 0

                new_pot += amount

            except Exception,e:

                print "Error: Bet amount must be > 0"

        return new_pot


    def get_opponent(self,player):

        if self.players[0] == player:

            return self.players[1]

        else:
            return self.players[0]