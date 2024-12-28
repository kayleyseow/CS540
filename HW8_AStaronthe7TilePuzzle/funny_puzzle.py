import heapq
import math

# Helper method to calculate the X position in a game
def getX(list, number):
    index = list.index(number)
    return index % 3

# Helper method to calculate the Y position in a game
def getY(list, number):
    index = list.index(number)
    return math.trunc(index/3)

def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    distance = 0;
    for i in range(len(to_state)):
        if from_state[i] == 0:
            continue
        distance = distance + abs(getX(from_state, from_state[i]) - getX(to_state, from_state[i])) 
        distance = distance + abs(getY(from_state, from_state[i]) - getY(to_state, from_state[i]))
    return distance

def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))

def getXFromIndex(index):
    return index % 3

def getYFromIndex(index):
    return math.trunc(index/3)

def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
    emptyTiles = []
    succ_states = []

    for i in range(len(state)):
        if state[i] == 0:
            emptyTiles.append(i)
    
    new_succ = state.copy()
    for i in range(len(emptyTiles)):
        x = getXFromIndex(emptyTiles[i])
        if (x-1 >= 0) and (new_succ[emptyTiles[i]-1] != 0): #checking move left
            new_succ[emptyTiles[i]] = new_succ[emptyTiles[i]-1]
            new_succ[emptyTiles[i]-1] = 0
            succ_states.append(new_succ)
            new_succ = state.copy()

        if (x+1 <= 2) and (new_succ[emptyTiles[i]+1] != 0): #checking move right
            new_succ[emptyTiles[i]] = new_succ[emptyTiles[i]+1]
            new_succ[emptyTiles[i]+1] = 0
            succ_states.append(new_succ)
            new_succ = state.copy()

        y = getYFromIndex(emptyTiles[i])
        if (y-1 >= 0) and (new_succ[emptyTiles[i]-3] != 0): #checking move up
            new_succ[emptyTiles[i]] = new_succ[emptyTiles[i]-3]
            new_succ[emptyTiles[i]-3] = 0
            succ_states.append(new_succ)
            new_succ = state.copy()

        if (y+1 <= 2) and (new_succ[emptyTiles[i]+3] != 0): #checking move down
            new_succ[emptyTiles[i]] = new_succ[emptyTiles[i]+3]
            new_succ[emptyTiles[i]+3] = 0
            succ_states.append(new_succ)
            new_succ = state.copy()   
    return sorted(succ_states)


def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """
    maxLength = 0
    priorityQueue = []
    trackPath = []
    closedSet = set()

    hscore = get_manhattan_distance(state)
    numberOfMoves = 0
    heapq.heappush(priorityQueue, (hscore, state, (numberOfMoves, hscore, -1)))

    while (len(priorityQueue) != 0):
        if (len(priorityQueue) > maxLength):
            maxLength = len(priorityQueue)
        
        temp = heapq.heappop(priorityQueue)
        tempstate = temp[1]
        trackPath.append(temp)
        if (tempstate == goal_state):
            break
        closedSet.add(tuple(temp[1]))
        numberOfMoves = temp[2][0] + 1


        successors = get_succ(tempstate)
        for state in successors:
            if tuple(state) in closedSet:
                continue
            hscore = get_manhattan_distance(state)
            
            heapq.heappush(priorityQueue, (hscore + numberOfMoves, state, (numberOfMoves, hscore, len(trackPath)-1)))
            

    last = temp 
    nextIndex = 0
    actualPath = []

    while (nextIndex != -1):
        nextIndex =  last[2][2]
        actualPath.insert(0, last)
        last = trackPath[nextIndex]

    for i in range(len(actualPath)):
        node = actualPath[i]
        print(node[1], "h={}".format(node[2][1]), "moves: {}".format(node[2][0]))
    print("Max queue length: {}".format(maxLength))

if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    print_succ([2,5,1,4,0,6,7,0,3])
    print()

    print(get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    print()

    solve([4, 3, 0, 5, 1, 6, 7, 2, 0])
    print()
