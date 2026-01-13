# Repository Structure and Data Description

A ML framework to predict the electronic forces of the square-lattice Holstein model coupled to two electrodes.

-------------------------------------------------
Training Data
-------------------------------------------------
The training_data directory contains two zip folders:

t0_547.zip: snapshots from time step 0 to time step 547

t547_1094.zip: snapshots from time step 547 to time step 1094

Each configuration is named cq*.dat, where * denotes the time step.

File format (cq*.dat):

Column 1: x-coordinate

Column 2: y-coordinate

Column 3: local electron density

Column 6: local lattice displacement (distortion) 

All other columns are not used in this project

The system is defined on a 40 × 30 square lattice.

-------------------------------------------------
Code
-------------------------------------------------

The code directory contains:

Training script for the machine-learning force-field model

Script for nonequilibrium dynamical simulations using the trained model

-------------------------------------------------
Neighbor Information
-------------------------------------------------

The neighbor directory contains CSV files specifying the neighbor indices for each lattice site, which are used to construct local environments for the machine-learning model.
