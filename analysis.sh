#!/bin/bash

#default settings
TARGET_DIR="output/sanmiguel_32spp_tau0.60" jupyter nbconvert --execute --to html --output "sanmiguel_32spp_tau0.60.html" --template full notebooks/analysis.ipynb 
TARGET_DIR="output/livingroom_32spp_tau1.00" jupyter nbconvert --execute --to html --output "livingroom_32spp_tau1.00.html" --template full notebooks/analysis.ipynb 
TARGET_DIR="output/twoboxes_32spp_tau0.14" jupyter nbconvert --execute --to html --output "twoboxes_32spp_tau0.14.html" --template full notebooks/analysis.ipynb 
