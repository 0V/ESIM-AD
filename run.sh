#!/bin/bash

# default settings
python main.py --data_root ./data/SanMiguel/ --spp 32 --ksize 13 --thres 0.6 --wlr simple
python main.py --data_root ./data/LivingRoom/ --spp 32 --ksize 13 --thres 1.0 --wlr simple
python main.py --data_root ./data/TwoBoxes/ --spp 32 --ksize 13 --thres 0.14 --wlr simple

