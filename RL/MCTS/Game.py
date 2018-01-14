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

    def assert_leaf(self,current_node,action):

        action_type = action.keys()[0]

        if current_node.action == action and action_type == "check":
            return True

        elif action_type == "call":
            return True

        elif action_type == "fold":
            return True

        else:
            return False

    def set_root(self,player,init_p1_cip,init_p2_cip):

        new_node = Node.Node(player,p1_cip=init_p1_cip,p2_cip=init_p2_cip,node_index=0)

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

        p1_cip,p2_cip = self.get_new_cip(current_node,player,action)

        is_leaf = self.assert_leaf(current_node,action)

        opponent = self.get_opponent(player)

        new_node = Node.Node(opponent,parent=current_node,action=action,p1_cip=p1_cip,p2_cip=p2_cip,is_leaf=is_leaf)

        self.tree.add_node(new_node)

    def get_new_cip(self,node,player,action):

        '''
        Return new pot size based off a new action
        :param pot:
        :param action:
        :return:
        '''

        action_type = action.keys()[0]

        amount = action.values()[0]

        p1_cip = node.p1_cip

        p2_cip = node.p2_cip

        if action_type in  ("bet","raise","call"):

            try:
                assert amount > 0

                if player == "p1":

                    p1_cip += amount

                else:
                    p2_cip += amount

            except Exception,e:

                print "Error: Bet amount must be > 0"

        return [p1_cip,p2_cip]

    def get_opponent(self,player):

        if self.players[0] == player:

            return self.players[1]

        else:
            return self.players[0]