# Hybrid Quantum Classical Graph Neural Networks for Particle Track Reconstruction

<p align="center">
  <img src="media/q_trkx_logo.png">
</p>

## How to use?

First, please refer to our [installation guide](docs/Installation.md)
to setup the necessary tools.

Use [```train.py```](./train.py) to train a model. 

Models are available in [```qnetworks```](./qnetworks) folder.

Choose the model and other hyperparameters using a configuration
file (see [```configs```](./configs) folder for examples).

Execute the following to train a model. 

```bash
python3 train.py [PATH-TO-CONFIG-FILE] 1 
```

or use the following to train multiple instances in parallel.

```bash
source send_jobs_multiple.sh [PATH-TO-CONFIG-FILE] [NUM_RUNS]
```

## Talks and Publications:

The list will be updated soon.
