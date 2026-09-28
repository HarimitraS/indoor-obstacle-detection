import cv2
import numpy as np

from coi_geometry import get_polygons


BACKGROUND = 6


def create_occupancy_map(
    cells,
    threshold=0.5
):

    polygons = get_polygons()

    occupancy = np.zeros(
        (480, 848),
        dtype=np.uint8
    )

    blocked_cells = []

    for cell_id, polygon in enumerate(polygons):

        probabilities = cells[cell_id]

        obstacle_scores = probabilities[
            :BACKGROUND
        ]

        best_class = np.argmax(
            obstacle_scores
        )

        best_score = obstacle_scores[
            best_class
        ]

        background_score = probabilities[
            BACKGROUND
        ]

        # Obstacle only if:
        #
        # obstacle probability is high
        # AND higher than background

        if (
            best_score >= threshold
            and
            best_score > background_score
        ):

            blocked_cells.append(
                cell_id
            )

            cv2.fillPoly(
                occupancy,
                [polygon],
                255
            )

    return occupancy, blocked_cells