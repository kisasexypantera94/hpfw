# hpfw

## Requirements
* Eigen3
* FFTW
* TBB
* essentia
* cereal
* MKL

## Running 
To configure:

```bash
cmake -S . -B build
```

To run examples
```bash
cmake --build build --target live-id 
./build/examples/live-id
```

## References
* Tsai, T. (2016). Audio Hashprints: Theory & Application. (Doctoral dissertation, EECS Department, University of California, Berkeley).


