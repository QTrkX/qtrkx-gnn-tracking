# Installation Instructions

## Install conda
You can use conda to manage all the dependencies. Please click [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) install conda.

## Setup the environment

After installing conda you need to setup an environment and then install
the dependencies. Please follow the instructions below. This might take some 
time.

```bash
git clone https://github.com/QTrkX/qtrkx-gnn-tracking.git
cd qtrkx-gnn-tracking
conda env create --file environment.yml
conda activate qtrkx
```

## Test the model

Now you are ready to run some QGNN models. You can run the command below to
test if everything is in order. This shouldn't take more than a few minutes
ideally. 

```bash
python train.py configs/test.yaml 1
```

If you see the AUC increasing after each test, everyhing should work like a charm. Enjoy!

