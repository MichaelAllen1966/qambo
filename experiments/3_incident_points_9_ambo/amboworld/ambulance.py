import random

class Ambulance():
    """
    Class for an individual ambulance.

    Parameters
    ----------
    _env (object):
        Reference to environment object

    ambulance_id (int):
        Unique ambulance ID
    assigned_dispatch_point (int):
        Currently assigned dispatch point
    at_dispatch_point (bool):
        Ambulance free and at dispatch point
    free (bool):
        Ambulance is currently free and may attend patient incidents
    travelling_to_disptach point (bool)
        Ambulance is free and travelling to dispatch point
    travelling_to_hospital (bool):
        Travelling to hospital with patient on board
    travelling_to_patient (bool):
        Travelling to patient incident

    Methods
    -------
    __init__():
        Constructor method

    """

    # Count of ambulances
    ambulance_count = 0

    def __init__(self, env,ambulance_id):
        """
        Constructor class for individual ambulance. Start at different random points each time
        """

        self._env = env
        self.ambulance_id = ambulance_id
        # Set starting position (random allocation)
        random.seed()
        self.dispatch_point = random.randint(0, self._env.number_dispatch_points -1)
        self.dispatch_x = self._env.dispatch_points[self.dispatch_point][0]
        self.dispatch_y = self._env.dispatch_points[self.dispatch_point][0]
        self.current_x = self.dispatch_x
        self.current_y = self.dispatch_y
        # Set starting status
        self.free = True
        self.at_hospital = False
        self.at_dispatch_point = True
        self.time_journey_start = None
        self.travelling_to_dispatch_point = False
        self.travelling_to_hospital = False
        self.travelling_to_patient = False
        self.journey_time = None
        self.start_x = None
        self.start_y = None
        self.target_x = None
        self.target_y = None