def get_neighbour_cells(cell, grid_r, grid_c, include_horizontal=True):
    row_cell, col_cell = cell
    neighbour_cells = []

    if row_cell > 0:  # Up
        to_be_appended_cell = (row_cell - 1, col_cell)
        neighbour_cells.append(to_be_appended_cell)
    if row_cell < grid_r - 1:  # Down
        to_be_appended_cell = (row_cell + 1, col_cell)
        neighbour_cells.append(to_be_appended_cell)
    if col_cell > 0:  # Left
        to_be_appended_cell = (row_cell, col_cell - 1)
        neighbour_cells.append(to_be_appended_cell)
    if col_cell < grid_c - 1:  # Right
        to_be_appended_cell = (row_cell, col_cell + 1)
        neighbour_cells.append(to_be_appended_cell)

    if include_horizontal:

        # we are not at first row, we can go up
        if row_cell > 0:
            # 2 ways up-right & up-left
            if col_cell < grid_c - 1:  # we can go up-right
                to_be_appended_cell = (row_cell - 1, col_cell + 1)
                neighbour_cells.append(to_be_appended_cell)
            if col_cell > 0:  # we aren't at first col so can go up-left
                to_be_appended_cell = (row_cell - 1, col_cell - 1)
                neighbour_cells.append(to_be_appended_cell)
        # we are not at last row, so can go down
        if row_cell < grid_r - 1:
            # 2 ways down-right & down-left
            if col_cell < grid_c - 1:  # we can go down-right
                to_be_appended_cell = (row_cell + 1, col_cell + 1)
                neighbour_cells.append(to_be_appended_cell)
            if col_cell > 0:  # we aren't at first col so can go down-left
                to_be_appended_cell = (row_cell + 1, col_cell - 1)
                neighbour_cells.append(to_be_appended_cell)

    return neighbour_cells

def get_neighbour_cells_optimized(cell, grid_r, grid_c, include_diagonals=True):
    row_cell, col_cell = cell
    neighbour_cells = []

    # cardinal directions
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # diagonals if requested
    if include_diagonals:
        directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dr, dc in directions:
        new_r, new_c = row_cell + dr, col_cell + dc
        if 0 <= new_r < grid_r and 0 <= new_c < grid_c:
            neighbour_cells.append((new_r, new_c))

    return neighbour_cells