# Multi-Objective Evolutionary Optimization of Virtualized Fast Feedforward Networks

This is the repo associated to the paper `Multi-Objective Evolutionary Optimization of Virtualized Fast Feedforward Networks`, to be published at EvoStar 2025.

To reproduce the experiments, you can run:

Virtualization via NSGA-II:
```bash
python src/main.py --multirun task=mnist,har,sc exp=movirt 
```
Virtualizatin via random search:
```bash
python src/main.py --multirun task=mnist,har,sc exp=randomvirt 
```
Landscape/Virtualization via manual tuning:
```bash
python src/main.py --multirun task=mnist,har,sc exp=weightvirt exp.page_size=0,1,2,3,4 exp.amount=0.5,0.6,0.7,0.8,0.9 
```
Note that these scripts run multiple experiments. If you want to run a single experiment, you can run:
```bash
python src/main.py task=task-name exp=experiment-name 
```
- Options for task-name: mnist, har, sc
- Options for expriment-name: train, movirt, randomvirt, weightvirt, generate_fim
  
Please see YAML files inside conf directory to see more details regarding configurations.
Code, data (including pretrained models and their corresponding fisher information matrices), and results are available at: https://dagshub.com/berab/MOE-VFFF.
