import queue
import string
from queue import Queue

import numpy as np

test_maze = np.zeros((5, 7))
blocks = [(0, 3), (1, 1), (1, 5), (2, 1), (3, 2), (3, 2), (4, 0), (4, 2), (4, 5)]
for block in blocks:
    test_maze[block] = 1


def get_accessible_neighbours(maze: np.ndarray, cell:tuple, grid_size:tuple, include_diagonals=True):
    """
    Returns a list of accessible neighboring cells from a given cell in a maze.

    Args:
        maze (np.ndarray): A 2D NumPy array representing the maze grid.
                           Cells with value 0 are accessible; cells with value 1 are obstacles.
        cell (tuple): The current cell as a tuple of (row, column).
        grid_size (tuple): The size of the maze grid as (rows, columns).
        include_diagonals (bool, optional): Whether to include diagonal neighbors. Defaults to True.

    Returns:
        list: A list of accessible neighboring cells as (row, column) tuples.
    """
    row_cell, col_cell = cell
    neighbour_cells = []
    grid_r, grid_c = grid_size
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

def reconstruct_shortest_path(parent: dict, start_cell: tuple, target_cell: tuple) -> list:
    """
    Reconstructs the shortest path from a start node to a target node using a parent mapping.

    Args:
        parent (dict): A dictionary mapping each node to its predecessor in the path.
        start_cell (str): The starting node of the path.
        target_cell (str): The target node to reach.

    Returns:
        list: A list of nodes representing the shortest path from start to target.
    """
    # path is initially empty
    path = []
    # we are currently at target, we start reversed (target -> start)
    current = target_cell

    while current != start_cell:
        # add current to the path
        path.append(current)
        # get the parent of the current
        current = parent[current]

    # we add start to the path finally
    path.append(start_cell)
    # reverse to resemble the path from start -> end
    path.reverse()
    return path

def bfs(start:tuple, graph:dict):
    """
    Performs a Breadth-First Search (BFS) on the given graph starting from the specified node.

    Tracks the traversal path by recording each node's predecessor (parent) in the search.

    Args:
        start (tuple): The starting node for BFS.
        graph (dict): A dictionary representing the graph where keys are node labels and values are lists of neighboring nodes.

    Returns:
        dict: A mapping of each visited node to its predecessor in the BFS traversal.
              Format: {child_node: parent_node}
    """

    visited = set()
    queue = Queue()
    queue.put(start)
    visited.add(start)
    prev = {}
    while not queue.empty():
        node = queue.get()
        for neighbour in graph[node]:
            if neighbour not in visited:
                queue.put(neighbour)
                visited.add(neighbour)
                prev[neighbour] = node

    return prev


def build_graph_from_matrix(matrix:np.ndarray):
    tree = {}

    for row in range(len(matrix)):
        for column in range(len(matrix[0])):
            cell = (row, column)
            tree[cell] = get_accessible_neighbours(matrix, cell, (len(matrix), len(matrix[0])))

    return tree



"""
The plan now is:
1- form a tree/graph using get_accessible_neighbours function from matrix/maze
2- preform bfs while recording prev cell using bfs function
3- find the shortest path from maze start to end from the prev dict from previous step
"""

"""
  ↓
[[0. 0. 0. 1. 0. 0. 0.]
 [0. 1. 0. 0. 0. 1. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0.]
 [1. 0. 1. 0. 0. 1. 0.]]
              ↓
Test maze looks like this where (0,0) is the start, (4,4) is the end, 1 = wall, 0 = part of the route

"""
print(test_maze)
graph = build_graph_from_matrix(test_maze)
child_parent = bfs((0,0), graph)
shortest_path = reconstruct_shortest_path(child_parent, (0,0), (4,4))
print("shortest path is\n", shortest_path) #  [(0, 0), (0, 1), (1, 2), (2, 2), (3, 3), (4, 4)]


