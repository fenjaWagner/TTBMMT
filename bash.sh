#!/bin/bash

for k in $(seq 1 2); do
    echo "Running with $k threads"
    for i in $(seq 0 1); do
        for j in $(seq 0 2); do
            echo "Running program with i=$i and j=$j (threads=$k)"
            OMP_NUM_THREADS=$k python3 experiments_threads.py "$i" "$j" "$k"
        done
    done
done
