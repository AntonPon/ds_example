import numpy as np
from collections import deque
import math


def explore_map(start_position: tuple[int, int], island_map: np.ndarray, m: int, n: int):
    queue = deque([start_position])
    while queue:
        i, j = queue.popleft()

        if i + 1 < m and island_map[i+1][j] == 1:
            island_map[i+1][j] = 0
            queue.append((i+1, j))
        if j + 1 < n and island_map[i][j+1] == 1:
            island_map[i][j+1] = 0
            queue.append((i, j+1))
        if i - 1 >= 0 and island_map[i-1][j] == 1:
            island_map[i-1][j] = 0
            queue.append((i-1, j))
        if j-1 >= 0 and island_map[i][j-1] == 1:
            island_map[i][j-1] = 0
            queue.append((i, j-1))


def island_counter(m: int, n: int, island_map: np.ndarray) -> int:
    island_number = 0
    for i in range(m):
        for j in range(n):
            if island_map[i][j] == 1:
                island_number += 1
                island_map[i][j] = 0
                explore_map((i, j), island_map, m, n)
    return island_number


if __name__ == '__main__':
    islnd_map = np.array([[1,1,1],[0,1,0],[1,1,1]])
    print(islnd_map)
    m = 3
    n = 3
    islnd_nmbr = island_counter(m, n, islnd_map)
    print(islnd_nmbr)
    islnd_map = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
    print(islnd_map)
    m = 3
    n = 4
    islnd_nmbr = island_counter(m, n, islnd_map)
    print(islnd_nmbr)
    islnd_map = np.array([[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1]])
    print(islnd_map)
    m = 3
    n = 4
    islnd_nmbr = island_counter(m, n, islnd_map)
    print(islnd_nmbr)