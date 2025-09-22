"""
A*
Heuristic calculation:
1. Manhattan Distance
 - When: Only moving in 4 directions, simple grid-based problems
 - Formula: h(n) = |x1 - x2| + |y1 - y2|

2. Chebyshev Distance
 - When: Moving in all 8 directions (with diagonals) where diagonal distance = 1
   like chess moves
 - Formula: h(n) = max(|x1 - x2|, |y1 - y2|)

3. Euclidean Distance
 - When: Moving in any direction (with diagonals) where diagonal distance is sqrt(2)
   like pythagorean theorem in real world and physics
 - Formula: h(n) = sqrt( (x1-x2)^2 + (y1-y2)^2 )
-----------------------------------------------------------------
The plan now is:
Suppose we have A (start) and P (end), then
1- Calculate the heuristic for A (based on any distance according to the problem, if we say that we have a grid
that allows diagonal then we are going with Chehyshev calculation, however we could also calculate the heuristic
for all, but let's just do it when needed instead)
2- calculate f = g(distance of node from start, A) + h (heuristic from prev calculation)
3- record previous node of the current node
4- look for current node neighbours and record their g, calculate their h, and compute f, then record A,
add A to closed/visited set, and its neighbours to open set
5- repeat by choosing the least value of f from the open set items, and if f value was found lower for a node in
open/visited set then its g, f, previous values are changed accordingly
6- stop once we reach discover node P (after removing it from open set), print the f for end node, and print the path by
repeatedly printing previous node for the current
"""
from queue import PriorityQueue

import numpy as np


def get_accessible_neighbours(maze: np.ndarray, cell: tuple, include_diagonals=True):
    """
    Returns a list of accessible neighboring cells from a given cell in a maze.

    Args:
        maze (np.ndarray): A 2D NumPy array representing the maze grid.
                           Cells with value 0 are accessible; cells with value 1 are obstacles.
        cell (tuple): The current cell as a tuple of (row, column).
        include_diagonals (bool, optional): Whether to include diagonal neighbors. Defaults to True.

    Returns:
        list: A list of accessible neighboring cells as (row, column) tuples.
    """
    row_cell, col_cell = cell
    neighbour_cells = []
    grid_r, grid_c = maze.shape
    # cardinal directions
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # diagonals if requested
    if include_diagonals:
        directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dr, dc in directions:
        new_r, new_c = row_cell + dr, col_cell + dc
        # if the cell is within maze matrix range, and the cell is not an obstacle (not 1)
        if 0 <= new_r < grid_r and 0 <= new_c < grid_c and maze[(new_r, new_c)] == 0:
            neighbour_cells.append((new_r, new_c))

    return neighbour_cells


maze = np.zeros((10, 10), dtype=int)
walls = [
    (1, 1), (1, 2), (1, 3), (1, 6), (1, 7), (1, 8),
    (2, 1), (2, 8),
    (3, 1), (3, 3), (3, 4), (3, 5), (3, 6), (3, 8),
    (4, 1), (4, 6),
    (5, 1), (5, 2), (5, 3), (5, 6), (5, 7), (5, 8),
    (6, 3),
    (7, 0), (7, 1), (7, 3), (7, 5), (7, 6), (7, 7), (7, 9),
    (8, 7),
    (9, 1), (9, 2), (9, 3), (9, 7)
]
for wall in walls:
    maze[wall] = 1

"""         Current maze
         0 1 2 3 4 5 6 7 8 9
0        . . . . . . . . . .
1        . 1 1 1 . . 1 1 1 .
2        . 1 . . . . . . 1 .
3        . 1 . 1 1 1 1 . 1 .
4        . 1 . . . . 1 . . .
5        . 1 1 1 . . 1 1 1 .
6        . . . 1 . . . . . .
7        1 1 . 1 . 1 1 1 . 1
8        . . . . . . . 1 . .
9        . 1 1 1 . . . 1 . P (end)
          A
        (start)
"""


def calculate_heuristic(cell, end_cell):
    """
    Calculates the heuristic of a cell based on Chehyshev Distance
    :param cell: current cell
    :param end_cell: goal cell
    :return: Chehyshev Distance Heuristic
    """
    return max(abs(cell[0] - end_cell[0]), abs(cell[1] - end_cell[1]))


start = (9, 0)
goal = (9, 9)


# print(h(start, goal))

def reconstruct_path(came_from, current):
    """
    Reconstructs path from dictionary holding (current_node, parent_node) pairs
    :param came_from: dict holding nodes
    :param current: last node at path
    :return: path from start to end
    """
    path = [current]
    while came_from[current] is not None:
        current = came_from[current]
        path.append(current)
    return list(reversed(path))


def A_star(start, goal, maze):
    """Finds the shortest path using A* algorithm"""
    # queue holds priority of (f_score, (cell))
    opened = PriorityQueue()
    # to record closed cells, to avoid rediscovery
    closed = set()
    g = dict() # records distance from current to start
    f = dict() # records f score for all nodes
    h = dict() # records heuristic for all nodes, could be cached
    came_from = dict() # records parent of the current node for all

    # init all for start node
    g[start] = 0
    h[start] = calculate_heuristic(start, goal)
    f[start] = h[start] + g[start]
    came_from[start] = None
    opened.put((f[start], start))

    while not opened.empty():
        f_score, current = opened.get()

        # if we reached goal then stop & reconstruct path
        if current == goal:
            return reconstruct_path(came_from, current)

        closed.add(current)
        neighbours = get_accessible_neighbours(maze, current)
        for neighbour in neighbours:
            # if we already discovered it, skip
            if neighbour in closed: continue
            # calculating before assigning
            # weights/distance from A to current node
            # since this is a maze/matrix/grid then we are saying
            # it costs 1 unit to move from 1 cell to another adjacent cell
            # so from A->B->C is 2 since 0 + 1 + 1
            current_g = g[current] + 1
            current_h = calculate_heuristic(neighbour, goal)
            current_f = current_h + current_g
            # if we didn't get better f score value for current node, do not assign
            if f.get(neighbour, float('inf')) <= current_f: continue

            # assign if we didn't do before, or if we got better f score value
            g[neighbour] = current_g
            h[neighbour] = current_h
            f[neighbour] = current_f
            came_from[neighbour] = current
            opened.put((f[neighbour], neighbour))


print(A_star(start, goal, maze))
