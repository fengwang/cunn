## CUNN -- Deep neural network with multi-GPU support writtend in a minimal fashion

#### Typical usage:

```cpp
    std::vector<unsigned long> dim{ 2, 20, 20, 1 };

	// construct NN
    auto nn = make_nn( dim );

	// train NN
	unsigned long const epoch = 1024;
    train( nn, "input.txt", "output.txt", epoch );

	// Validate
    double res = validate( nn, "in.txt", "out.txt" );

    // Predict
    auto out = predict( nn, "input.txt" );

	// save NN
    nn.save_as("xor.nn");

	// load NN
    auto nm = load_nn("xor.nn");
```

#### Typical compilation and link command (MAC OS X) :

```
clang++ -c -std=c++17 -stdlib=libc++ -O3 -ferror-limit=2 -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-sign-conversion -Wno-exit-time-destructors -Wno-float-equal -Wno-global-constructors -Wno-missing-declarations -Wno-unused-parameter -Wno-padded -Wno-shadow -Wno-weak-vtables -Wno-missing-prototypes -Wno-unused-variable -ferror-limit=1 -Wno-deprecated -Wno-conversion -Wno-double-promotion -fPIC -Wno-documentation -Wno-old-style-cast -Wno-reserved-id-macro -Wno-documentation-unknown-command -Wno-undef -Iinclude -Icumar/include -I/Developer/NVIDIA/CUDA-9.0/include -o ./obj/cuda.o src/cuda.cc
clang++ -c -std=c++17 -stdlib=libc++ -O3 -ferror-limit=2 -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-sign-conversion -Wno-exit-time-destructors -Wno-float-equal -Wno-global-constructors -Wno-missing-declarations -Wno-unused-parameter -Wno-padded -Wno-shadow -Wno-weak-vtables -Wno-missing-prototypes -Wno-unused-variable -ferror-limit=1 -Wno-deprecated -Wno-conversion -Wno-double-promotion -fPIC -Wno-documentation -Wno-old-style-cast -Wno-reserved-id-macro -Wno-documentation-unknown-command -Wno-undef -Iinclude -Icumar/include -I/Developer/NVIDIA/CUDA-9.0/include -o ./obj/cumar.o cumar/src/cumar.cc
clang++ -c -std=c++17 -stdlib=libc++ -O3 -ferror-limit=2 -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-sign-conversion -Wno-exit-time-destructors -Wno-float-equal -Wno-global-constructors -Wno-missing-declarations -Wno-unused-parameter -Wno-padded -Wno-shadow -Wno-weak-vtables -Wno-missing-prototypes -Wno-unused-variable -ferror-limit=1 -Wno-deprecated -Wno-conversion -Wno-double-promotion -fPIC -Wno-documentation -Wno-old-style-cast -Wno-reserved-id-macro -Wno-documentation-unknown-command -Wno-undef -Iinclude -Icumar/include -I/Developer/NVIDIA/CUDA-9.0/include -o ./obj/xor_test.o test/xor_test.cc
clang++ -o ./bin/xor_test ./obj/xor_test.o ./obj/cuda.o ./obj/cumar.o -lc++ -lc++abi -O3 -lcurand -lcudart -lnvrtc -lnvblas -lcublas -L/Developer/NVIDIA/CUDA-9.0/lib -framework CUDA
```

