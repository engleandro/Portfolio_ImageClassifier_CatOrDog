#!/usr/bin/bash

mkdir dataset
cd dataset

wget -P . https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
tar -xf images.tar.gz -C .

cd ..
