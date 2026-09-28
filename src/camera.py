import cv2
import numpy as np

from obstacle_detector import detect
from occupancy_map import create_occupancy_map
from path_planner import plan_path
from controller import NavigationController
from coi_geometry import get_polygons


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = 0.50

camera = cv2.VideoCapture(0)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    848
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


if not camera.isOpened():

    print("ERROR: Camera not found.")

    exit()


controller = NavigationController()

polygons = get_polygons()


print()
print("==============================")
print("INDOOR NAVIGATION SYSTEM")
print("==============================")
print()
print("Q = quit")
print()


while True:

    ret, frame = camera.read()

    if not ret:

        break


    # ========================================================
    # 1. AI DETECTION
    # ========================================================

    cells = detect(frame)


    # ========================================================
    # 2. OCCUPANCY MAP
    # ========================================================

    occupancy, blocked_cells = (
        create_occupancy_map(
            cells,
            CONFIDENCE_THRESHOLD
        )
    )


    # ========================================================
    # 3. PATH PLANNING
    # ========================================================

    path = plan_path(
        occupancy
    )


    # ========================================================
    # 4. CONTROLLER
    # ========================================================

    action = controller.update(
        path
    )


    # ========================================================
    # 5. DRAW OBSTACLES
    # ========================================================

    for cell_id in blocked_cells:

        cv2.polylines(
            frame,
            [
                polygons[cell_id]
            ],
            True,
            (0, 0, 255),
            3
        )


        # Cell number

        M = cv2.moments(
            polygons[cell_id]
        )

        if M["m00"] != 0:

            cx = int(
                M["m10"] /
                M["m00"]
            )

            cy = int(
                M["m01"] /
                M["m00"]
            )

            cv2.putText(
                frame,
                f"OBS {cell_id}",
                (
                    cx - 30,
                    cy
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                2
            )


    # ========================================================
    # 6. DRAW PATH
    # ========================================================

    if path is not None:

        for i in range(
            len(path) - 1
        ):

            p1 = path[i]

            p2 = path[i + 1]

            cv2.line(
                frame,
                p1,
                p2,
                (255, 0, 0),
                4
            )


    # ========================================================
    # 7. START POINT
    # ========================================================

    start = (
        frame.shape[1] // 2,
        frame.shape[0] - 10
    )

    cv2.circle(
        frame,
        start,
        8,
        (0, 255, 0),
        -1
    )


    # ========================================================
    # 8. STATUS PANEL
    # ========================================================

    if blocked_cells:

        status = (
            f"OBSTACLE: "
            f"{len(blocked_cells)} CELLS"
        )

    else:

        status = "PATH CLEAR"


    cv2.rectangle(
        frame,
        (10, 10),
        (420, 115),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        status,
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"ACTION: {action}",
        (25, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"BLOCKED: {blocked_cells}",
        (25, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 255),
        1
    )


    # ========================================================
    # 9. SHOW
    # ========================================================

    cv2.imshow(
        "AI Indoor Obstacle Avoidance",
        frame
    )


    # ========================================================
    # 10. QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


camera.release()

cv2.destroyAllWindows()