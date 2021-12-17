#!/bin/bash

N=20

for i in {1..20}
do
    echo "Starting job number $i out of $N"
    qsub job9
done