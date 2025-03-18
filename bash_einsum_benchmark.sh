#!/bin/bash



for i in $(seq 0 9); do
    for j in $(seq 0 3); do
        echo "Running program with $i and $j"
        python3 e_b.py "$i" "$j"
    done
done
