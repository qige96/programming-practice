import copy
import a_star

def select(fringe):
    return fringe.get()

def g(node, goal_state):
    return node.steps

def h(node, goal_state):
    """Hamming distance"""
    h = 0
    for i in range(0, 3):
        for j in range(0, 3):
            if goal_state[i][j] != node.state[i][j] and node.state[i][j] != 0:
                h += 1
    return h


def get0index(node):
    for i in range(0,3):
        for j in range(0,3):
            if node.state[i][j] == 0:
                return (i, j)


def move(direction, node):
    x, y = get0index(node)
    new_node = a_star.SearchNode(node.state)
    new_node.prev = node
    new_node.steps += 1
    if direction == 'up':
        new_node.state[x][y] = new_node.state[x - 1][y]
        new_node.state[x - 1][y] = 0
    elif direction == 'down':
        new_node.state[x][y] = new_node.state[x + 1][y]
        new_node.state[x + 1][y] = 0
    elif direction == 'left':
        new_node.state[x][y] = new_node.state[x][y - 1]
        new_node.state[x][y - 1] = 0
    elif direction == 'right':
        new_node.state[x][y] = new_node.state[x][y + 1]
        new_node.state[x][y + 1] = 0
    else:
        raise ValueError("direction incorrect!")
    return new_node  

def expand(problem, node):
    x, y = get0index(node)
    successors = []
    if x > 0:
        successors.append(move('up', node))
    if x < 2:
        successors.append(move('down', node))
    if y > 0:
        successors.append(move('left', node))
    if y < 2:
        successors.append(move('right', node))
    return successors

def judge(state):
    Onedomain=[]
    for r in state:
        for each in r:
            if each!=0:
                Onedomain.append(each)
    count=0
    for i in range(1,len(Onedomain)):
        for j in range(i):
            if Onedomain[j]>Onedomain[i]:
                count+=1
    return count%2==0

if __name__ == '__main__':
    init_state = [[1,3,4],[8,0,5],[7,2,6]]
    goal_state = [[1,2,3],[8,0,4],[7,6,5]]
    if judge(init_state) != judge(goal_state):
        print("No solutions!")
    else:
        problem = a_star.Problem(init_state, goal_state)
        print(a_star.AStarTreeSearch(problem, g, h, select, expand))

