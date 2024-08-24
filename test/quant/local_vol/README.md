This project can be built independently:

```
$ makdir -p build 
$ cmake . -B build -GNinja -DCMAKE_C_COMPILER=<clang path> -DCMAKE_CXX_COMPILER=<clang++ path> -DLLVM_DIR=<llvm path> -DCMAKE_BUILD_TYPE=Release
$ cmake --build build
$ build/local_vol_test
```