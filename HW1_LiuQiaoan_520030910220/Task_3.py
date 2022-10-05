import DR20API
import numpy as np
from queue import PriorityQueue
import random
from matplotlib import pyplot as plt
import scipy.special
### START CODE HERE ###
MAX_G_COST = 1e3
q = PriorityQueue()
Manhattan_or_Euclidean = 2 # 1 means Manhattan and 2 means Euclidean
random.seed("AI3603")
def h(pos, goal_pos):
    return np.linalg.norm(np.array(pos)-np.array(goal_pos),ord=Manhattan_or_Euclidean)
def obstacles_cost(current_map, current_pos, direc):
    """
    Given current map of the world, direction of the robot, calculate the obstacles cost for each of 8 directions.

    Arguments:
    current_map -- A 120*120 array indicating current map, where 0 indicating traversable and 1 indicating obstacles.
    current_pos -- A 2D vector indicating the current position of the robot.
    direc -- A 2*1 array indicationg the pos after one of [U,UR,R,DR,D,DL,L,UL] movement.

    Returns:
    A double
    """
    distance = 3
    width = 5
    left_right = direc[0]
    up_down = direc[1]
    center = [current_pos[0]+distance*left_right,current_pos[1]+distance*up_down]
    return 1/32*np.sum(current_map[center[0]-width:center[0]+width+1,center[1]-width:center[1]+width+1])

class node:
    def __init__(self, pos, goal_pos):
        self.pos = pos
        self.is_visited = False
        self.ancester = pos
        self.h_cost = h(pos, goal_pos)+random.uniform(0, 3603e-6)
        self.g_cost = MAX_G_COST

    def cost(self):
        return self.h_cost+self.g_cost
    
    def __call__(self):
        self.f = self.h_cost+self.g_cost
        return self.f, self



def explore(node):
    
    current_x, current_y = node.pos
    ances_x, ances_y = node.ancester
    direc = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
    for i,(move_x, move_y) in enumerate(direc):
        if not (0 <= current_x+move_x <= 119 and 0 <= current_y+move_y <= 119):
            break
        if current_map[current_x+move_x][current_y+move_y]:
            break
        next_node = distance[current_x+move_x][current_y+move_y]
        steer_cost = np.linalg.norm([current_x-ances_x-move_x,current_y-ances_y-move_y],ord=1) * 4
        obstc_cost = obstacles_cost(current_map, node.pos , direc[i])
        
        if i % 2 == 0:
            if node.g_cost + 1 + steer_cost + obstc_cost < next_node.g_cost:
                next_node.g_cost = node.g_cost + 1 + steer_cost + obstc_cost
                next_node.ancester = node.pos
                q.put(next_node())
        else:
            if node.g_cost + np.sqrt(2) + steer_cost + obstc_cost < next_node.g_cost:
                next_node.g_cost = node.g_cost + np.sqrt(2) + steer_cost + obstc_cost
                next_node.ancester = node.pos
                q.put(next_node())

def planner(current_map, current_pos, goal_pos):
    """
    Given current map of the world, current position of the robot and the position of the goal, 
    plan a path from current position to the goal using improved A* algorithm.

    Arguments:
    current_map -- A 120*120 array indicating current map, where 0 indicating traversable and 1 indicating obstacles.
    current_pos -- A 2D vector indicating the current position of the robot.
    goal_pos -- A 2D vector indicating the position of the goal.

    Return:
    path -- A N*2 array representing the planned path by improved A* algorithm.
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
    while x != current_pos[0] or y != current_pos[1]:
        x,y = distance[x][y].ancester
        path.append([x,y])
    path = path[-3::-1] if len(path) < 22 else path[-3::-1][:22]
    path = np.array(path)

    def bernstein_poly(n, i, t): 
        return scipy.special.comb(n, i) * t ** i * (1 - t) ** (n - i)
 
    def bezier(t, control_points): 
        n = len(control_points) - 1
        return np.sum([bernstein_poly(n, i, t) * control_points[i] for i in range(n + 1)], axis=0)
    traj = []
    n_points = 220
    for t in np.linspace(0, 1, n_points):
        traj.append(bezier(t, path))
    path = traj
    # print(f"path = {path}")
    ###  END CODE HERE  ###
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
    print(f"current_pos={current_pos}")
    current_map = controller.update_map()
    total_path = []
    # Plan-Move-Perceive-Update-Replan loop until the robot reaches the goal.
    while not reach_goal(current_pos, goal_pos):
        # Plan a path based on current map from current position of the robot to the goal.
        path = planner(current_map, current_pos, goal_pos)
        total_path.extend(path[0:len(path)-10])
        print(f"path = {path}")
        # Move the robot along the path to a certain distance.
        controller.move_robot(path)
        # Get current position of the robot.
        current_pos = controller.get_robot_pos()
        print(f"current_pos={current_pos}")
        # Update the map based on the current information of laser scanner and get the updated map.
        current_map = controller.update_map()

    # Stop the simulation.
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
    plt.savefig("task3.pdf")

###  END CODE HERE  ###