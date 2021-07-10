# qambo
Experiments in Deep Q Learning controlling ambulance placement

[![DOI](https://zenodo.org/badge/326734877.svg)](https://zenodo.org/badge/latestdoi/326734877)

### Introductory video

https://youtu.be/UYJtOLYcOU8

## Run on BinderHub:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MichaelAllen1966/qambo/main)

# General description

The simulation environment, using SimPy and OpenAI gym, provides an environment where:

* Incidents occurs in areas within a world with fixed dimensions. The geographic pattern of incidents may change throughout the day.

* When an incident occurs, ambulances are dispatched from fixed dispatch points; the closest free ambulance is used.

* The ambulance collects a patient and conveys them to the closest hospital.

* The ambulance is then allocated, by an agent, to any dispatch point. The ambulance travels to that dispatch point where it becomes available to respond to incidents (the ambulance may also be allocated while travelling to a dispatch point, depending on simulation environment settings).

* The job of the agent is to allocate ambulances to dispatch points in order to minimise the time from incident to arrival of ambulance at the scene of the incident.


## Abstract 

*Background and motivation:* Deep Reinforcement Learning  (Deep RL) is a rapidly developing field. Historically most application has been made to games (such as chess, Atari games, and go). Deep RL is now reaching the stage where it may offer value in real world problems, including optimisation of healthcare systems. One such problem is where to locate ambulances between calls in order to minimise time from emergency call to ambulance on-scene. This is known as the Ambulance Location problem.

*Aim:* To develop an OpenAI Gym-compatible framework and simulation environment for testing Deep RL agents.

*Methods*: A custom ambulance dispatch simulation environment was developed using OpenAI Gym and SimPy. Deep RL agents were built using PyTorch. The environment is a simplification of the real world, but allows control over the number of clusters of incident locations, number of possible dispatch locations, number of hospitals, and switching of incident locations throughout the day.

*Results*: A range of Deep RL agents based on Deep Q networks were tested in this custom environment. All reduced time to respond to emergency calls compared with random allocation to dispatch points. Bagging Noisy Duelling Deep Q networks gave the most consistence performance. All methods had a tendency to lose performance if trained for too long, and so agents were saved at their optimal performance (and tested on independent simulation runs).

*Conclusion*: Deep RL methods, developed using simulated environments, have the potential to offer a novel approach to optimise the Ambulance Location problem. Creating open simulation environments should allow more rapid progress in this field.

## Agents tested

* Random assignment.
    
* Double Deep Q Network (ddqn): Standard Deep Q Network, with policy and target networks.
    
* Duelling Deep Q Network (3dqn): Policy and target networks calculate Q from sum of *value* of state and *advantage* of each action (*advantage* represents the added value of an action compared to the mean value of all actions).
    
* Noisy Duelling Deep Q Network (noisy 3dqn). Networks have layers that add Gaussian noise to aid exploration.
    
* Prioritised Replay Duelling Deep Q Network (pr 3dqn). When training the policy network, steps are sampled from the memory using a method that prioritises steps where the network had the greatest error in predicting Q.
    
* Prioritised Replay Noisy Duelling Deep Q Network (pr noisy 3dqn). Combining prioritised replay with noisy layers.
    
* Bagging Deep Q Network (bagging ddqn), with 5 networks. Multiple networks are trained from different bootstrap samples from the memory. Action may be sampled at random from networks, or a majority vote used.
    
* Bagging Duelling Deep Q Network (bagging 3dqn), with 5 networks. Combining the bagging multi-network approach with the duelling architecture.
    
* Bagging Noisy Duelling Deep Q Network (bagging noisy 3dqn) with 5 networks. Combining the bagging multi-network approach with the duelling architecture and noisy layers.
    
* Bagging Prioritised Replay Noisy Duelling Deep Q Network (bagging pr noisy 3dqn) with 5 networks. Combining the bagging multi-network approach with the duelling architecture, noisy layers, and prioritised replay.


