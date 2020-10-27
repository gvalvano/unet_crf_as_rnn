#!/usr/bin/env bash

rm lattice_filter.so
mkdir build_dir
cd build_dir

# define compilers to use
CUDA_COMPILER=/usr/local/cuda/bin/nvcc
CXX_COMPILER=/usr/bin/g++-4.8
CUDA_INCLUDE=/usr/local/cuda/include/

## Parameters:
#SPATIAL_DIMS=2
#INPUT_CHANNELS=4
#REFERENCE_CHANNELS=1
#MAKE_TESTS=False

# get parameters from config file:
config_file='../config.txt'
# read file line by line; assign what is separated by IFS ("=") to "name" and "value"
while IFS="="
  read name value
    do
      if [ "$name" == "SPATIAL_DIMS" ]; then
          SPATIAL_DIMS="$value"; fi
      if [ "$name" == "INPUT_CHANNELS" ]; then
          INPUT_CHANNELS="$value"; fi
      if [ "$name" == "REFERENCE_CHANNELS" ]; then
          REFERENCE_CHANNELS="$value"; fi
      if [ "$name" == "MAKE_TESTS" ]; then
          MAKE_TESTS="$value"; fi
    done < $config_file

printf "\nBuilding...\n"
echo " | reading config from:" "$config_file"
echo " | parameters: $SPATIAL_DIMS", "$INPUT_CHANNELS", "$REFERENCE_CHANNELS", "$MAKE_TESTS"
echo " | make..."
echo " "

cmake -DCMAKE_BUILD_TYPE=Debug -D CMAKE_CUDA_COMPILER=${CUDA_COMPILER} \
                               -D CMAKE_CXX_COMPILER=${CXX_COMPILER} \
                               -D CMAKE_CUDA_HOST_COMPILER=${CXX_COMPILER} \
                               -D CUDA_INCLUDE=${CUDA_INCLUDE} \
                               -D SPATIAL_DIMS="${SPATIAL_DIMS}" \
                               -D INPUT_CHANNELS="${INPUT_CHANNELS}" \
                               -D REFERENCE_CHANNELS="${REFERENCE_CHANNELS}" \
                               -D MAKE_TESTS="${MAKE_TESTS}" \
                               -G "CodeBlocks - Unix Makefiles" ../
make
printf "Done.\n"

cp lattice_filter.so ../
cd ..
rm -r build_dir


