#!/bin/bash

version=('reduced' 'EFG')
len=(56 161 1081 2801 6881 10601)
ks=(5 8 10)

for v in {0..1}
do
    for i in {0..2}
    do
        python3 main.py "${version[$v]}" "${len[$i]}" "${k}"
    done
done