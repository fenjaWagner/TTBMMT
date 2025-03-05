#!/bin/bash



for i in $(seq 0 27); do
    for j in $(seq 0 3); do
        echo "Running program with $i and $j"
        python3 experiments_from_bash.py "$i" "$j"
    done
done
