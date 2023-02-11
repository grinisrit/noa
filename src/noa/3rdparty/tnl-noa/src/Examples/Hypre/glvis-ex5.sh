#!/bin/bash

ex=ex5
keys=Aaamc

dir="${1:-.}"
mesh="$dir/$ex.mesh"
sol="$dir/$ex.sol"

if ! test -e "$mesh"; then
    echo "Error: cannot find mesh file for $ex"
    exit 1
fi

echo "FiniteElementSpace" > "$sol"
echo "FiniteElementCollection: H1_2D_P1" >> "$sol"
echo "VDim: 1" >> "$sol"
echo "Ordering: 0" >> "$sol"
echo "" >> "$sol"
find "$dir" -name "$ex.sol.??????" | sort | xargs cat >> "$sol"

glvis -m "$mesh" -g "$sol" -k "$keys"
