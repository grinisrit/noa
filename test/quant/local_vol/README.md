## Differentiable Dupire Local Volatility Model

This project is based on [EnzymeAD](https://github.com/EnzymeAD/Enzyme) and requires `clang` and the `LLVM` toolchain to build.

The target can be built independently from main `NOA` library:

```
$ makdir -p build 
$ cmake . -B build -GNinja -DCMAKE_C_COMPILER=<clang path> -DCMAKE_CXX_COMPILER=<clang++ path> -DLLVM_DIR=<llvm path> -DCMAKE_BUILD_TYPE=Release
$ cmake --build build
$ build/local_vol_test
```