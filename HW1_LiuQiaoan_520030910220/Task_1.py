import DR20API
import numpy as np
import random
from queue import PriorityQueue
from matplotlib import pyplot as plt

### START CODE HERE ###
# This code block is optional. You can define your utility function and class in this block if necessary.
MAX_G_COST = 1e3
q = PriorityQueue()
Manhattan_or_Euclidean = 1 # 1 means Manhattan and 2 means Euclidean
random.seed("AI3603")
def h(pos, goal_pos):
    return np.linalg.norm(np.array(pos)-np.array(goal_pos),ord=Manhattan_or_Euclidean)

class node:
    def __init__(self, pos, goal_pos):
        self.pos = pos
        self.is_visited = False
        self.ancester = pos
        self.steer_cost = 0
        self.h_cost = h(pos, goal_pos)+random.uniform(0, 3603e-6)
        self.g_cost = MAX_G_COST

    def cost(self):
        return self.h_cost+self.g_cost
    
    def __call__(self):
        self.f = self.h_cost+self.g_cost+self.steer_cost
        return self.f, self



def explore(node):
    
    current_x, current_y = node.pos
    ances_x, ances_y = node.ancester
    direc = [(0,1), (0,-1), (-1,0), (1,0)]
    for move_x, move_y in direc:
        if not (0 <= current_x+move_x <= 119 and 0 <= current_y+move_y <= 119):
            break
        if current_map[current_x+move_x][current_y+move_y]:
            break
        next_node = distance[current_x+move_x][current_y+move_y]
        if node.g_cost + 1 < next_node.g_cost:
            next_node.g_cost = node.g_cost + 1
            next_node.steer_cost = np.linalg.norm([current_x-ances_x-move_x,current_y-ances_y-move_y],ord=1) * 4
            next_node.ancester = node.pos
            q.put(next_node())

###  END CODE HERE  ###

def A_star(current_map, current_pos, goal_pos):
    """
    Given current map of the world, current position of the robot and the position of the goal, 
    plan a path from current position to the goal using A* algorithm.

    Arguments:
    current_map -- A 120*120 array indicating current map, where 0 indicating traversable and 1 indicating obstacles.
    current_pos -- A 2D vector indicating the current position of the robot.
    goal_pos -- A 2D vector indicating the position of the goal.

    Return:
    path -- A N*2 array representing the planned path by A* algorithm.
    """

    ### START CODE HERE ###
    global distance
    distance = [[node([x,y],[100,100]) for y in range(0,120)] for x in range(0,120)]
    source_node = distance[current_pos[0]][current_pos[1]]
    source_node.g_cost = 0
    q.put(source_node())

    while not q.empty():
        current_node = q.get()[1]
        if np.abs(current_node.pos[0] - goal_pos[0]) < 3 and np.abs(current_node.pos[1] - goal_pos[1]) < 3:
            break
        if current_node.is_visited:
            continue
        explore(current_node)
        current_node.is_visited = True
    
    x,y = current_node.pos
    path=[[x,y]]
    while not (x == current_pos[0] and y == current_pos[1]):
        x,y = distance[x][y].ancester
        path.append([x,y])
    path = path[-3::-1] if len(path) < 32 else path[-3::-1][:32]

    ###  END CODE HERE  ###
    print(f"path = {path}")
    return path

def reach_goal(current_pos, goal_pos):
    """
    Given current position of the robot, 
    check whether the robot has reached the goal.

    Arguments:
    current_pos -- A 2D vector indicating the current position of the robot.
    goal_pos -- A 2D vector indicating the position of the goal.

    Return:
    is_reached -- A bool variable indicating whether the robot has reached the goal, where True indicating reached.
    """

    ### START CODE HERE ###
    if np.abs(current_pos[0] - goal_pos[0]) < 20 and np.abs(current_pos[1] - goal_pos[1]) < 20:
        return True
    else:
        return False


    ###  END CODE HERE  ###
    return is_reached

if __name__ == '__main__':
    # Define goal position of the exploration, shown as the gray block in the scene.
    goal_pos = [100, 100]
    controller = DR20API.Controller()

    # Initialize the position of the robot and the map of the world.
    current_pos = controller.get_robot_pos()
    print(f"current_pos = {current_pos}")
    current_map = controller.update_map()
    total_path = []
    # Plan-Move-Perceive-Update-Replan loop until the robot reaches the goal.
    while not reach_goal(current_pos, goal_pos):
        # Plan a path based on current map from current position of the robot to the goal.
        path = A_star(current_map, current_pos, goal_pos)
        total_path.extend(path[0:len(path)-3])
        # Move the robot along the path to a certain distance.
        controller.move_robot(path)
        # Get current position of the robot.
        current_pos = controller.get_robot_pos()
        print(f"current_pos = {current_pos}")
        # Update the map based on the current information of laser scanner and get the updated map.
        current_map = controller.update_map()

    # Plot and Stop the simulation.
    controller.stop_simulation()

    obstacles_x, obstacles_y = [], []
    for i in range(120):
        for j in range(120):
            if current_map[i][j] == 1:
                obstacles_x.append(i)
                obstacles_y.append(j)

    path_x, path_y = [], []
    for path_node in total_path:
        path_x.append(path_node[0])
        path_y.append(path_node[1])

    plt.plot(path_x, path_y, "-r")
    plt.plot(current_pos[0], current_pos[1], "xr")
    plt.plot(goal_pos[0], goal_pos[1], "xb")
    plt.plot(obstacles_x, obstacles_y, ".k")
    plt.grid(True)
    plt.axis("equal")
    plt.savefig("task1.pdf")