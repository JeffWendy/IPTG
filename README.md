# IPTG
A Deep Pedestrian Trajectory Generator for Complex Indoor Environments

The IPTG model itself, dataset buidling, training and evaluation related scripts are included in the model directory. The file train.py is the entry point of the 

Implementation of evaluation metrics such as bleu score, kl-divergence of feature nodes are included in the evaluation directory.

datapreprocessing directory contains scripts that perform transformation between raw csv records, pedestrain trajectories, feature sequences, and rasters. Some data mining code is also in this directory.

Implementation and training of baseline models are included in the baselines directory.

Data generated or utilized in the experiments are put in the top directory or data directory. 