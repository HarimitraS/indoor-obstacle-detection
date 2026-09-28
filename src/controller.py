from enum import Enum


class State(Enum):

    FORWARD = 1
    STOP = 2
    REPLAN = 3
    MOVE_LEFT = 4
    MOVE_RIGHT = 5


class NavigationController:

    def __init__(self):

        self.state = State.FORWARD

        self.current_path = None


    def update(
        self,
        path
    ):

        # ---------------------------------------------
        # NO PATH
        # ---------------------------------------------

        if path is None:

            self.state = State.STOP

            return "STOP"


        # ---------------------------------------------
        # PATH EXISTS
        # ---------------------------------------------

        self.current_path = path


        if len(path) < 2:

            self.state = State.FORWARD

            return "FORWARD"


        # ---------------------------------------------
        # Determine first meaningful waypoint
        # ---------------------------------------------

        start = path[0]

        waypoint = path[
            min(
                10,
                len(path) - 1
            )
        ]


        dx = (
            waypoint[0]
            - start[0]
        )


        # ---------------------------------------------
        # Direction
        # ---------------------------------------------

        if abs(dx) < 25:

            self.state = State.FORWARD

            return "FORWARD"


        if dx < 0:

            self.state = State.MOVE_LEFT

            return "TURN LEFT"


        self.state = State.MOVE_RIGHT

        return "TURN RIGHT"