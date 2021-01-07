import gym
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
import random
import simpy

from amboworld.ambulance import Ambulance
from amboworld.patient import Patient
from amboworld.utils import get_distance


class Env(gym.Env):
    """Custom Environment that follows gym interface.

    A discrete event simulation environment using SimPy

    Attributes
    ----------
    action_number (int):
        Number of possible actions
    action_space (gym space - discrete):
        Gym space listing discrete choice of dispatch points
    ambo_speed (float):
        Distance (km) covered in minute
    ambos_assigned_to_dispatch_points (NumPy array):
        Number of ambulances assigned to each dispatch points
    ambulances (list, ambulance objects):
        List of ambulance objects
    calls (int):
        Number of calls (demand)
    completed_incidents (list of patients):
        Completed patient objects
    demand_met(int):
        Number of patients ambulances respond to
    dispatch_points (list of tuples):
        List of x,y co-ordinate tuples of dispatch points
    free_ambulances (list):
        List of free ambulances
    hospitals (list of tuples):
        List of x,y co-ordinate tuples of hospitals points
    incident_frequency (float):
        Mean time between incidents
    incident_points (list):
        List of x,y co-ordinate tuples of centres of incident points
    incident_range (int):
        Limits of incidents around incident centre (plus/minus)
    max_size (int):
        Max x,y km co-ordinates
    number_ambulances (int):
        Number of ambulances in model
    number_dispatch_points (int):
        Number of dispatch points
    number_epochs (int):
        Number of epochs per day. Each epoch has a different set of incident
        locations. Epochs are equally spaced throughout day.
    number_hospitals (int):
        Number of hospitals
    number_incident_points (int):
        Number of centre points where incidents may occur. Random range added
    observation_size (int):
        Number of observations returned
    observation_space (gym space - box):
        Gym space providing n-dimensional array of observation space.
        Provides number of ambulances assigned to each dispatch point
    patients_waiting_for_assignment (list)
        List of patients waiting for ambulance assignment
    patients_assignment_to_ambo_arrival (list)
        List of patients waiting for ambulance arrival after assignment
    patients_in_transit (list)
        List of patients in transit to hospital
    random_seed (int):
        Random seed used to set hospital and dispatch point locations
    render_env (bool):
        Whether to render amboworld
    render_grid_size (int):
        Size (characters per line) of grid used to render environment
    render_interval (int):
        Time between rendering environment
    reward_response_times (list):
        List of ambulance response times in time between returning reward to
        agent
    results_assignment_to_ambo_arrival (list):
        List of all times from ambo assignment to arrival with patient
    results_call_to_ambo_arrival (list):
        List of all times from call to ambo arrival with patient
    sim_time_step (float):
        Simulation time step (minutes)
    unallocated_ambo (list):
        List of ambulances waiting allocations (occurs each sim step)


    Internal methods
    ----------------

    __init__:
        Constructor method
    _calculate_reward:
        Calculate reward
    _get_observations:
        Get current state observations
    _set_dispatch_points:
        Create list of x,y tuple dispatch points (using random uniform 
        distribution)
    _set_hospital_locations:
        Create list of x,y tuple hospitals (using random uniform distribution)



    External facing methods
    -----------------------

    close

    reset

    step

    render
    """

    def __init__(self, max_size=50, 
                number_ambulances=8,
                number_dispatch_points=25,
                number_epochs=2,
                number_incident_points=4,
                incident_range=0.0,
                number_hospitals=1,
                duration_incidents=1e5,
                ambo_kph=60.0,
                random_seed=42,
                incident_interval=20,
                time_step=1,
                render_env=False,
                print_output=False,
                render_grid_size=25,
                render_interval=10,
                ambo_free_from_hospital=True):
        """Constructor class for amboworld"""

        # Inherit from super class
        super(Env, self).__init__()
        # Set attributes

        self.action_number = int(number_dispatch_points)
        self.ambo_free_from_hospital = ambo_free_from_hospital
        self.ambo_speed = ambo_kph / 60
        self.ambos_assigned_to_dispatch_points = \
            np.zeros(number_dispatch_points)
        self.counter_patients = 0
        self.counter_ambulances = 0
        self.dispatch_points = []
        self.hospitals = []
        self.incident_interval = incident_interval
        self.incident_range = incident_range
        self.incident_points = []
        self.max_size = int(max_size)
        self.number_ambulances = max(1, int(number_ambulances))
        self.number_dispatch_points = max(1, int(number_dispatch_points))
        self.number_epochs = number_epochs
        self.number_hospitals = max(1, int(number_hospitals))
        self.number_incident_points = number_incident_points
        self.observation_size = number_dispatch_points + 3
        self.random_seed = int(random_seed)
        self.render_env = bool(render_env)
        self.render_grid_size = render_grid_size
        self.render_interval = render_interval
        self.sim_duration = duration_incidents
        self.sim_time_step = time_step
        self.step_count = 0

        # Set action space (as a choice from dispatch points)
        self.action_space = spaces.Discrete(self.number_dispatch_points)
        # Set observation space: number of ambos currently assigned to each 
        # dispatch point + Location of ambo to be assigned (as fraction 0-1),
        # and time of day (as fraction 0-1)
        self.observation_space = spaces.Box(
            low=0, high=number_ambulances - 1, 
            shape=(number_dispatch_points + 3, 1), dtype=np.uint8)

        # Set up hospital, dispatch  and incident point locations
        self._set_dispatch_points()
        self._set_incident_locations()
        self._set_hospital_locations()

        # During writing use printing of times
        self.print_output = print_output

    def _assign_ambo(self, patient):
        """
        Assign closest ambulance to patient
        """

        # Get closest ambulance
        best_distance = 9999999
        for ambo in self.free_ambulances:        
            if ambo.at_dispatch_point or ambo.at_hospital:
                ambo_x = ambo.current_x
                ambo_y = ambo.current_y
            else:
                # Check if ambo may be assigned before reaching dispatch point
                if self.ambo_free_from_hospital:
                # Need to work out current location
                    time_elapsed = self.simpy_env.now - ambo.time_journey_start
                    fraction_travelled = time_elapsed / ambo.journey_time
                    ambo_x = (ambo.start_x + ((ambo.target_x - ambo.start_x) *
                            fraction_travelled))
                    ambo_y = (ambo.start_y + ((ambo.target_y - ambo.start_y) *
                            fraction_travelled))
        
            # Get distance from patient to ambulance
            distance = get_distance(patient.incident_x, patient.incident_y,
                                    ambo_x, ambo_y)
            if distance < best_distance:
                # New best distance found
                best_distance = distance
                best_ambo = ambo

        # Best ambo identified
        ambo = best_ambo
        # Move patient between lists and remove ambulance from free ambos list
        self.patients_assignment_to_ambo_arrival.append(patient)
        self.free_ambulances.remove(ambo)
        ambo.travelling_to_patient = True
        ambo.target_x = patient.incident_x
        ambo.target_y = patient.incident_y
        ambo.start_x = ambo.current_x
        ambo.start_y = ambo.current_y

        # Set time and calculate time to arrival
        patient.time_ambo_assigned = self.simpy_env.now
        ambo_travel_time = best_distance / self.ambo_speed
        ambo.journey_time = ambo_travel_time
        ambo.time_journey_start = self.simpy_env.now
        ambo.free = False
        if self.print_output:
            print(f'Patient {patient.id} ambulance {ambo.ambulance_id} assigned: {self.simpy_env.now:0.1f}')
        # SimPy timeout for ambulance travel
        yield self.simpy_env.timeout(ambo_travel_time)

        # Ambo has arrived with patient
        self.demand_met += 1
        ambo.travelling_to_patient = False
        patient.time_ambo_arrive = self.simpy_env.now
        ambo.current_x = patient.incident_x
        ambo.current_y = patient.incident_y
        if self.print_output:
            print(f'Patient {patient.id} ambulance arrived: {self.simpy_env.now:0.1f}')
        self.patients_assignment_to_ambo_arrival.remove(patient)
        self.patients_in_transit.append(patient)
        assigned_to_arrival = \
            patient.time_ambo_arrive - patient.time_ambo_assigned
        self.results_assignment_to_ambo_arrival.append(assigned_to_arrival)
        call_to_arrival = patient.time_ambo_arrive - patient.time_call
        self.results_call_to_ambo_arrival.append(call_to_arrival)
        self.reward_response_times.append(assigned_to_arrival)

        # Get closest hospital and set travel to hospital
        ambo.travelling_to_hospital = True
        best_distance = 9999999
        for hospital_index in range(self.number_hospitals):
            distance = get_distance(
                patient.incident_x, patient.incident_y,
                self.hospitals[hospital_index][0], 
                self.hospitals[hospital_index][1])
            if distance < best_distance:
                # New best distance found
                best_distance = distance
                best_hospital = hospital_index
        patient.allocated_hospital = best_hospital
        ambo.target_x = self.hospitals[best_hospital][0]
        ambo.target_y = self.hospitals[best_hospital][1]

        ambo_travel_time = distance / self.ambo_speed
        # SimPy timeout for ambulance travel
        ambo.time_journey_start = self.simpy_env.now
        ambo.start_x = ambo.current_x
        ambo.start_y = ambo.current_y
        ambo.journey_time = ambo_travel_time
        yield self.simpy_env.timeout(ambo_travel_time)

        # Patient has arrived at hospital
        ambo.travelling_to_hospital = False
        ambo.current_x = self.hospitals[best_hospital][0]
        ambo.current_y = self.hospitals[best_hospital][1]
        if self.print_output:
            print(f'Patient {patient.id} arrived at hospital: {self.simpy_env.now:0.1f}')
        patient.time_arrive_at_hospital = self.simpy_env.now
        self.completed_incidents.append(patient)

        # Reset ambulance to wait for new dispatch point
        self.free_ambulances.append(ambo)
        ambo.time_journey_start = None
        ambo.journey_time = None
        ambo.start_x = None
        ambo.start_y = None
        ambo.at_hospital = True
        self.ambos_assigned_to_dispatch_points[ambo.dispatch_point] -= 1
        ambo.dispatch_point = None
        self.unallocated_ambos.append(ambo)

    def _calculate_reward(self):
        """
        Calculate reward
        """

        # Reward is negative of time to respond. Use oldest time each steo
        reward = None
        if len(self.reward_response_times) > 0:
            reward = 0 - self.reward_response_times.pop(0)
            
        reward = 0 - reward ** 2

        return reward

    def _check_for_unassigned_patients(self):
        """
        Each minute check for unassigned patients and free ambulances
        """
        while True:
            yield self.simpy_env.timeout(1)
            patients_waiting = len(self.patients_waiting_for_assignment)
            ambos_free = len(self.free_ambulances)
            patients_to_assign = min(patients_waiting, ambos_free)
            if patients_to_assign > 0:
                for i in range(patients_to_assign):
                    patient = self.patients_waiting_for_assignment.pop(0)
                    self.simpy_env.process(self._assign_ambo(patient))

    def _generate_demand(self):
        """
        Generate demand (in an infinite loop)
        """

        while True:
            # Sample time to next incident
            time_out = random.expovariate(1 / self.incident_interval)
            yield self.simpy_env.timeout(time_out)
            # Generate patient
            self.calls += 1
            self.counter_patients += 1
            patient = Patient(self.simpy_env, self.counter_patients, 
                self.number_incident_points, self.incident_points, 
                self.incident_range, self.max_size, self.number_epochs)
            self.patients_waiting_for_assignment.append(patient)
            if self.print_output:
                print(f'Incident: {self.simpy_env.now:0.1f}')

    def _get_observations(self, ambo):
        """
        Return observations, including location of ambulance to assign dispatch
        point
        """

        # Get assigned dispatch points
        obs = list(self.ambos_assigned_to_dispatch_points)
        # Get x and y of ambo to be assigned
        x = ambo.current_x / self.max_size
        y = ambo.current_y / self.max_size
        # Get time of day (0-1)
        day = int(self.simpy_env.now / 1440)
        time = self.simpy_env.now - (day * 1440)
        time = time / 1440
        obs.extend([x, y, time])

        obs = np.array(obs)

        return obs

    def _set_dispatch_points(self):
        """
        Set ambulance dispatch points using uniform random distribution
        """

        # Get number of rows and cols to populate
        rows_cols = int(self.number_dispatch_points ** 0.5)
        add_random = self.number_dispatch_points - (rows_cols ** 2)
        # Padding before any dispatch point is placed
        pad = self.max_size / (rows_cols + 1) /2
        points = np.linspace(pad, self.max_size - pad, rows_cols)
        for x in points:
            for y in points:
                self.dispatch_points.append((x, y))

        # Add any 'extra points' as random points
        if add_random > 0:
            random.seed(self.random_seed + 2)
            for _ in range(self.add_random):
                x = random.uniform(0, self.max_size)
                y = random.uniform(0, self.max_size)
                self.dispatch_points.append((x, y))

    def _set_hospital_locations(self):
        """
        Set hospital locations. Use set location sup to 4 hospitals,
        othwerwise use uniform random distribution
        """
        
        if self.number_hospitals == 1:
            x = self.max_size / 2
            y = self.max_size / 2
            self.hospitals.append((x, y))
        elif self.number_hospitals == 2:
            x = 1/3 * self.max_size
            y = 1/3 * self.max_size
            self.hospitals.append((x, y))
            x = 2/3 * self.max_size
            y = 2/3 * self.max_size
            self.hospitals.append((x, y))
        elif self.number_hospitals == 3:
            x = 1/2 * self.max_size
            y = 1/3 * self.max_size
            self.hospitals.append((x, y))
            x = 1/3 * self.max_size
            y = 2/3 * self.max_size
            self.hospitals.append((x, y))
            x = 2/3 * self.max_size
            y = 2/3 * self.max_size
            self.hospitals.append((x, y))
        elif self.number_hospitals == 4:
            x = 1/4 * self.max_size
            y = 1/4 * self.max_size
            self.hospitals.append((x, y))
            x = 3/4 * self.max_size
            y = 1/4 * self.max_size
            self.hospitals.append((x, y))
            x = 1/4 * self.max_size
            y = 3/4 * self.max_size
            self.hospitals.append((x, y))
            x = 3/4 * self.max_size
            y = 3/4 * self.max_size
            self.hospitals.append((x, y))
        else:
            random.seed(self.random_seed)
            for _ in range(self.number_hospitals):
                x = random.uniform(0, self.max_size)
                y = random.uniform(0, self.max_size)
                self.hospitals.append((x, y))

    def _set_incident_locations(self):
        """
        Set centres of incident locations. Each epoch in a day has a different
        set of incident locations.
        """
        random.seed(self.random_seed + 1)
        for epcoh in range(self.number_epochs):
            epoch_incident_points = []
            for _ in range(self.number_incident_points):
                x = random.uniform(0, self.max_size)
                y = random.uniform(0, self.max_size)
                epoch_incident_points.append((x, y))
            self.incident_points.append(epoch_incident_points)

    def _travel_to_dispatch_point(self, ambo):

        """
        Ambulance travel to dispatch point
        """

        # Ambulance travels to dispatch point
        ambo.travelling_to_dispatch_point = True
        ambo.at_hospital = False
        ambo.target_x = ambo.dispatch_x
        ambo.target_y = ambo.dispatch_y
        # Get time to dispatch point
        distance = get_distance(
            ambo.current_x, ambo.current_y, ambo.dispatch_x, ambo.dispatch_y)
        ambo_travel_time = distance / self.ambo_speed
        ambo.time_journey_start = self.simpy_env.now
        ambo.journey_time = ambo_travel_time
        ambo.start_x = ambo.current_x
        ambo.start_y = ambo.current_y
        yield self.simpy_env.timeout(ambo_travel_time)

        # Ambulance at dispatch point (end of sequence)
        if self.print_output:
            print(f'Ambulance at dispatch point {self.simpy_env.now:0.1f}')
        ambo.current_x = ambo.dispatch_x
        ambo.current_y = ambo.dispatch_y
        ambo.start_x = None
        ambo.start_y = None
        ambo.target_x = None
        ambo.target_y = None
        ambo.travelling_to_dispatch_point = False
        ambo.at_dispatch_point = True
        ambo.free = True
        ambo.time_journey_start = None
        ambo.journey_time = None

    def close(self):
        """
        Clean up any necessary simulation objects, e.g. display
        """

        del self.simpy_env

        return 0

    def render(self):
        """
        Render environment info in terminal

        x = dispatch point
        ! = patient waiting for ambo
        a = free ambulance
        A = non-free ambulance

        """
        while True:

            # Delay between rendering environment
            yield self.simpy_env.timeout(self.render_interval)

            # Set up empty grid
            grid = np.chararray((self.render_grid_size, self.render_grid_size), 
                unicode=True)
            grid.fill('.')

            # Calculate scaling factor for sim co-ordinates to render grid
            scale = self.render_grid_size / self.max_size

            # Add dispatch points as x
            for dispatch_point in self.dispatch_points:
                x = int(dispatch_point[0] * scale)
                y = int(dispatch_point[1] * scale)
                grid[y][x] = 'x'

            # Add hospitals as H
            for hospital in self.hospitals:
                x = int(hospital[0] * scale)
                y = int(hospital[1] * scale)
                grid[y][x] = 'H'

            # Add patients waiting for ambo as !
            for patient in self.patients_waiting_for_assignment:
                x = int(patient.incident_x * scale)
                y = int(patient.incident_y * scale)
                grid[y][x] = '!'

            # Loop through Ambos
            for ambo in self.ambulances:
                if ambo.free:
                    x = int(ambo.current_x * scale)
                    y = int(ambo.current_y * scale)
                    grid[y][x] = 'a'
                elif ambo.at_hospital:
                    x = int(ambo.current_x * scale)
                    y = int(ambo.current_y * scale)
                    grid[y][x] = 'A'
                else:
                    # Need to work out current location
                    time_elapsed = self.simpy_env.now - ambo.time_journey_start
                    fraction_travelled = time_elapsed / ambo.journey_time
                    x = (ambo.start_x + ((ambo.target_x - ambo.start_x) * 
                            fraction_travelled))
                    y = (ambo.start_y + ((ambo.target_y - ambo.start_y) * 
                            fraction_travelled))
                    x = int(x * scale)
                    y = int(y * scale)
                    grid[y][x] = 'A'

            # Print grid (unpack rows with *)
            for row in grid:
                print(*row)

            print()

        print('Stop')

    def reset(self):
        """
        Reset method:
        1) Create new SimPy environment
        2) Start SimPy processes (incidents)
        3) Pick one ambo to be assigned to a dispatch point
        4) Pass back first observations
        """

        self.simpy_env = simpy.Environment()

        # Reset lists and trackers
        self.completed_incidents = []
        self.free_ambulances = []
        self.unallocated_ambos = []
        self.patients_waiting_for_assignment = []
        self.patients_assignment_to_ambo_arrival = []
        self.patients_in_transit = []
        self.results_call_to_ambo_arrival = []
        self.results_assignment_to_ambo_arrival = []
        self.reward_response_times = []
        self.calls = 0
        self.demand_met = 0
        self.step_count = 0

        # Set up ambulances
        self.ambulances = []
        for i in range(self.number_ambulances):
            ambo = Ambulance(self, i)
            self.ambulances.append(ambo)
            self.free_ambulances.append(ambo)
            self.ambos_assigned_to_dispatch_points[ambo.dispatch_point] += 1

        # Start continuous processes
        self.simpy_env.process(self._generate_demand())
        self.simpy_env.process(self._check_for_unassigned_patients())
        if self.render_env:
            self.simpy_env.process(self.render())
        # Make one ambulance waiting for assignement
        ambo = self.free_ambulances.pop()
        ambo.at_dispatch_point = False
        ambo.free = False
        self.unallocated_ambos.append(ambo)
        self.ambos_assigned_to_dispatch_points[ambo.dispatch_point] -= 1
        ambo.dispatch_point = None
        obs = self._get_observations(self.unallocated_ambos[0])

        return obs

    def step(self, action):
        """
        Simulation step.
        1) Update dispatch points for each ambulance. Note that this will only
        change ambulance behaviour once an ambulance has taken a patient to 
        hospital and then travels to assigned dispatch point
        """

        self.step_count += 1

        # Assign dispatch point to ambulance
        ambo = self.unallocated_ambos.pop(0)
        dispatch_point = int(action)
        self.ambos_assigned_to_dispatch_points[dispatch_point] += 1
        ambo.dispatch_point = dispatch_point
        ambo.dispatch_x = self.dispatch_points[dispatch_point][0]
        ambo.dispatch_y = self.dispatch_points[dispatch_point][1]
        # Pass to travel to dispatch point process
        process = self._travel_to_dispatch_point(ambo)
        self.simpy_env.process(process)

        # Keep steeping through simulation until ambulance free for allocation
        while len(self.unallocated_ambos) == 0:
            self.simpy_env.run(until=self.simpy_env.now + self.sim_time_step)

        # Get return values
        reward = self._calculate_reward()
        terminal = True if self.step_count >= self.sim_duration else False
        obs = self._get_observations(self.unallocated_ambos[0])
        info = {
            'response_times': self.results_assignment_to_ambo_arrival,
            'call_to_arrival': self.results_call_to_ambo_arrival,
            'assignment_to_arrival': self.results_assignment_to_ambo_arrival,
            'calls': self.calls,
            'fraction_demand_met': np.round(self.demand_met / self.calls, 3)}

        return (obs, reward, terminal, info)
