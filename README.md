# Generalization bounds for neural ordinary differential equations and deep residual networks

## Environment

### With conda

```
conda env create -f environment.yml
```

### With pip

Install Python 3.9.9 and pip 22.1.2, then

```
pip3 install -r requirements.txt
```

## Reproducing the paper figures

Takes about 60 hours to run on a standard laptop (no GPU).
To shorten the run time, diminish the number of epochs and/or of repetitions in the config.py file.

```
python main.py
```
