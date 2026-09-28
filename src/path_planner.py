import cv2
import heapq
import numpy as np


def find_nearest_free(
    occupancy,
    point,
    radius=50
):

    x, y = point

    height, width = occupancy.shape

    for r in range(radius):

        for dx, dy in [
            (r, 0),
            (-r, 0),
            (0, r),
            (0, -r)
        ]:

            nx = x + dx
            ny = y + dy

            if (
                0 <= nx < width
                and
                0 <= ny < height
            ):

                if occupancy[ny, nx] == 0:

                    return (
                        nx,
                        ny
                    )

    return None


def heuristic(a, b):

    return abs(
        a[0] - b[0]
    ) + abs(
        a[1] - b[1]
    )


def astar(
    occupancy,
    start,
    goal,
    scale=8
):

    small = cv2.resize(
        occupancy,
        (
            occupancy.shape[1] // scale,
            occupancy.shape[0] // scale
        ),
        interpolation=cv2.INTER_NEAREST
    )

    start = (
        start[0] // scale,
        start[1] // scale
    )

    goal = (
        goal[0] // scale,
        goal[1] // scale
    )

    h, w = small.shape

    if not (
        0 <= start[0] < w
        and
        0 <= start[1] < h
    ):
        return None

    if not (
        0 <= goal[0] < w
        and
        0 <= goal[1] < h
    ):
        return None

    # Find free start
    if small[start[1], start[0]] != 0:

        result = find_nearest_free(
            small,
            start,
            10
        )

        if result is None:
            return None

        start = result

    # Find free goal
    if small[goal[1], goal[0]] != 0:

        result = find_nearest_free(
            small,
            goal,
            10
        )

        if result is None:
            return None

        goal = result

    open_set = []

    heapq.heappush(
        open_set,
        (
            0,
            start
        )
    )

    came_from = {}

    g_score = {
        start: 0
    }

    directions = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1)
    ]

    while open_set:

        _, current = heapq.heappop(
            open_set
        )

        if current == goal:

            path = []

            while current in came_from:

                path.append(current)

                current = came_from[
                    current
                ]

            path.append(start)

            path.reverse()

            return [
                (
                    x * scale,
                    y * scale
                )
                for x, y in path
            ]

        for dx, dy in directions:

            nx = current[0] + dx
            ny = current[1] + dy

            if not (
                0 <= nx < w
                and
                0 <= ny < h
            ):
                continue

            if small[ny, nx] != 0:
                continue

            neighbor = (
                nx,
                ny
            )

            tentative_g = (
                g_score[current]
                + 1
            )

            if (
                neighbor not in g_score
                or
                tentative_g
                < g_score[neighbor]
            ):

                came_from[
                    neighbor
                ] = current

                g_score[
                    neighbor
                ] = tentative_g

                f_score = (
                    tentative_g
                    + heuristic(
                        neighbor,
                        goal
                    )
                )

                heapq.heappush(
                    open_set,
                    (
                        f_score,
                        neighbor
                    )
                )

    return None


def plan_path(occupancy):

    height, width = occupancy.shape

    # Robot/camera position
    start = (
        width // 2,
        height - 10
    )

    # Desired forward direction
    goal = (
        width // 2,
        20
    )

    path = astar(
        occupancy,
        start,
        goal
    )

    return path