# Structural Inspection Planner

This repository was made for a research project on how to perform an automated, visual inspection. See the [paper](Strutural%20Inspection%20Planner%20Report.pdf) for details. 

Run `viewpoint_generation.py` script to see the algorithm in action. The script contains many functions but the basic recipe is detailed below.

Given a mesh file, compute a set of viewpoints that see the entire object using the `dual_viewpoint_sampling` function. For each unseen vertex belonging to the object's mesh, randomly sample a viewpoint within a cone and check if it's reachable from the ground's free configuration space which is modeled as a Probabilistic Road Map (PRM). Loop until all vertices are seen.  

With the viewpoint set, find the smallest number of viewing positions reachable from one point on the ground. Hierarchial clustering performs the optimization.

A final touch solves the shortest path between all the ground view positions via the Traveling Salesman Problem (TSP).
