#!/bin/bash

for i in {1..10}
do
    cpu p train.py -r 0 -m 1 -ro >> results.txt
    cpu p train.py -r 1 -m 1 -ro >> results.txt
    cpu p train.py -r 2 -m 1 -ro >> results.txt

    cpu p train.py -r 0 -m 1 -b -ro >> results.txt
    cpu p train.py -r 1 -m 1 -b -ro >> results.txt
    cpu p train.py -r 2 -m 1 -b -ro >> results.txt

    cpu p train.py -r 0 -m 0 -ro >> results.txt
    cpu p train.py -r 1 -m 0 -ro >> results.txt
    cpu p train.py -r 2 -m 0 -ro >> results.txt

done
