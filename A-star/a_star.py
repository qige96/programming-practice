
import math
from queue import PriorityQueue


class Problem:
    def __init__(self, i_state, g_state):
        self.initial_state = i_state
        self.goal_state = g_state

    def is_goal(self, state):
        return self.goal_state == state

class SearchNode:
    def __init__(self, state=None, prev=None, steps=0, evaluation=math.inf):
        self.state = state
        self.prev = prev
        self.steps = steps
        self.evaluation = evaluation 

    def __str__(self):
        return str(self.state)

    def __repr__(self):
        return self.__str__()
    
    def __lt__(self, other):
        return self.evaluation < other.evaluation

    def __eq__(self, other):
        return self.state == other.state

def path_to(node):
    path = []
    ptr_node = node
    while ptr_node:
        path.append(ptr_node)
        ptr_node = ptr_node.prev
    path.reverse()
    return path


def AStarTreeSearch(problem, g, h, select, expand):
    fringe = PriorityQueue()
    init_node = SearchNode(problem.initial_state)
    fringe.put(init_node)

    while True:
        if fringe.empty():
            raise Exception("Failure")
        node = select(fringe)
        if problem.is_goal(node.state):
            return path_to(node)
        for successor in expand(problem, node):
            successor.evaluation = g(successor, problem.goal_state) + h(successor, problem.goal_state)
            fringe.put(successor)

