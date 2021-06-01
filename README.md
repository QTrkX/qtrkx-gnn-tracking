# Hybrid Quantum Classical Graph Neural Networks for Particle Track Reconstruction

<p align="center">
  <img src="media/q_trkx_logo.png">
</p>

## How to use?

Use [```train.py```](./train.py) to train a model. Models are available in [```qnetworks```](./qnetworks) folder. Choose the model and other hyperparameters using a configuration file (see [```configs```](./configs) folder for examples).

Execute the following to train the model. 

``` python3 train.py [PATH-TO-CONFIG-FILE] ```

or use the following to train multiple instances in parallel.

``` source send_jobs_multiple.sh [PATH-TO-CONFIG-FILE] [NUM_RUNS] ```

## Publication list:
