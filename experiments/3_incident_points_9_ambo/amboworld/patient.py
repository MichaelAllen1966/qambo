import numpy as np
import random

class Patient():
    """
    Patient class. Tacks patient calls.

    Attributes
    ----------
    id (int):
        Unique patient id
    incident_x (float):
        x location of call
    incident_y (float):
        y location of call
    time_ambo_arrive (float):
        Sim time that ambo arrives
    time_ambo_assigned (float):
        Sim time that ambo is assigned to patient
    time_arrive_at_hospital (float):
        Sim time that patient arrives at hospital
    time_call (float):
        Sim time that patient calls for ambo

    Methods
    _______
    __init__():
        Constructor method
    incident:
        Incident pathway
    """

    def __init__(self, env, patient_id, number_incident_points, incident_points,
                 incident_range, max_size, number_epochs):


        # Set link to SimPy environment
        self._env = env

        # Set id
        self.id = patient_id

        # Set trackers
        self.time_call = self._env.now
        self.time_ambo_assigned = None
        self.time_ambo_arrive = None
        self.time_arrive_at_hospital = None
        self.allocated_hospital = None

        # Get epoch
        day = int(self._env.now / 1440)
        time_of_day = self._env.now - (day * 1440)
        epoch_length = 1440 / number_epochs
        epoch = int (time_of_day / epoch_length)

        # Set incident location based on incident centres and random jitter
        incident_point = random.randint(0, number_incident_points - 1)
        self.incident_x = incident_points[epoch][incident_point][0]
        self.incident_y = incident_points[epoch][incident_point][1]
        self.incident_x += random.uniform(-incident_range, incident_range)
        self.incident_y += random.uniform(-incident_range, incident_range)
        self.incident_x = np.clip(self.incident_x, 0, max_size)
        self.incident_y = np.clip(self.incident_y, 0, max_size)









