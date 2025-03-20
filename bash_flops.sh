#!/bin/bash

for backend in $(seq 0 3); do
    for method in $(seq 0 3); do
        for (( size=5; size<=50; size+=5 )); do
            echo "Running program with $backend $method $size"
            python3 experiments_flops.py "$backend" "$method" "$size"
        done
    done
done

for backend in $(seq 0 3); do
    for method in $(seq 0 3); do
        for (( size=53; size<=59; size+=2 )); do
            echo "Running program with $backend $method $size"
            python3 experiments_flops.py "$backend" "$method" "$size"
        done
    done
done