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

for i in $(seq 0 9); do
    for j in $(seq 0 3); do
        echo "Running program with $i and $j"
        python3 e_b.py "$i" "$j"
    done
done


for i in $(seq 0 9); do
    for j in $(seq 0 3); do
        echo "Running program with $i and $j"
        python3 e_b_double.py "$i" "$j"
    done
done

python3 plots_dict.py
