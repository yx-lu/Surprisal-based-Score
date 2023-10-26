# [NeurIPS2023] Calibrating “Cheap Signals” in Peer Review without a Prior

## Description

This code repository contains the numerical experiments carried out in the paper, including the plotting of the experimental result images.

## Run code

### Requirements

The python file runs on `Python 3.7`, including `matplotlib==3.5.1`, `numpy==1.21.5` and `scipy==1.7.3`. See `requirements.txt`. 

### Run command

The cached experimental data is saved in the `./save` folder, and the process of generating experimental data is deterministic. To use the cached experimental data to draw figures, type `python ./paint.py` in the terminal. The code can be regenerate the experimental data by modifying the `load` parameter to `calc`.  Generating the experimental data takes about half a day.