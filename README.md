# Digital Modelling Of Protein Systems Using Monte-Carlo Simultaion

This repository containt the implementation for a monte-carlo particle simulation model.

It currently supports the [canonical](https://en.wikipedia.org/wiki/Canonical_ensemble)
and [grand canonical](https://en.wikipedia.org/wiki/Grand_canonical_ensemble) ensembles

The implementation includes support for particles of different types,
and with different amount of patches, though it only includes the Kern-Frenkel model for patch-patch interaction.

For hard-sphere interaction, the implementation includes a square-well potential and yukawa potential.

## Dependencies

To run this program, you would need to have the cuda runtime installed.
Since we do not provide binaries, you would also likely need the code toolkit to compile the program.

Both could be installed from https://developer.nvidia.com/cuda-toolkit

Make sure that you also have proper hardware and drivers installed.

## Compilation

This compilation is straightforward 

1. Make sure you have `cmake >= 3.15`

2. ```sh 
cmake -B build
cmake --build build -j$(nproc)
```
<!---->
<!-- ## Usage -->
<!---->
<!-- Just specify the number of particles  -->
<!---->
<!-- ```sh  -->
<!---->
<!-- ``` -->
