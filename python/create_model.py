

def grid_to_xy(grid, feature_indices):
    X = [[None for _ in range(len(grid[0]))] for _ in range(len(grid))]
    Y = [[None for _ in range(len(grid[0]))] for _ in range(len(grid))]

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if not grid[i][j] is None:
                X[i][j] = [grid[i][j]['X'][e] for e in feature_indices]
                Y[i][j] = grid[i][j]['Y']
    return X, Y

