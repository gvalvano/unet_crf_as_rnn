#!/usr/bin/env bash

rm lattice_filter.so
rm lattice_filter_op_loader.py
cd permutohedral_lattice
bash build.sh
cp lattice_filter.so ../
cp lattice_filter_op_loader.py ../
rm lattice_filter.so

