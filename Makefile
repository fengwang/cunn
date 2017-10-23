CXX           = clang++
#CXXFLAGS        = -std=c++1z -stdlib=libc++ -O3 -ferror-limit=2 -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-sign-conversion -Wno-exit-time-destructors -Wno-float-equal -Wno-global-constructors -Wno-missing-declarations -Wno-unused-parameter -Wno-padded -Wno-shadow -Wno-weak-vtables -Wno-missing-prototypes -Wno-unused-variable -ferror-limit=1 -Wno-deprecated -Wno-conversion -Wno-double-promotion -fPIC -Wno-documentation -Wno-old-style-cast -Wno-reserved-id-macro -Wno-documentation-unknown-command -Wno-undef -DDEBUGLOG
CXXFLAGS        = -std=c++17 -stdlib=libc++ -O3 -ferror-limit=2 -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-sign-conversion -Wno-exit-time-destructors -Wno-float-equal -Wno-global-constructors -Wno-missing-declarations -Wno-unused-parameter -Wno-padded -Wno-shadow -Wno-weak-vtables -Wno-missing-prototypes -Wno-unused-variable -ferror-limit=1 -Wno-deprecated -Wno-conversion -Wno-double-promotion -fPIC -Wno-documentation -Wno-old-style-cast -Wno-reserved-id-macro -Wno-documentation-unknown-command -Wno-undef
INCPATH       = -Iinclude -Icumar/include -I/Developer/NVIDIA/CUDA-9.0/include
LINK          = $(CXX)
LFLAGS        = -lc++ -lc++abi -O3 -lcurand -lcudart -lnvrtc -lnvblas -lcublas -L/Developer/NVIDIA/CUDA-9.0/lib -framework CUDA      ## mac config
#LFLAGS        = -lc++ -lc++abi -O3 -lcurand -lcudart -lnvrtc -lcuda -lcublas -L/opt/cuda/lib64 -L/usr/local/cuda-7.0/lib64 ## linux config
DEL_FILE      = rm -f

####### Output directory
OBJECTS_DIR   = ./obj
BIN_DIR       = ./bin
LIB_DIR       = ./lib

all: cunn

clean:
	rm -rf $(OBJECTS_DIR)/*.o
	rm -rf $(BIN_DIR)/*
	rm -rf $(LIB_DIR)/*.a

cuda.o: src/cuda.cc
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o $(OBJECTS_DIR)/cuda.o src/cuda.cc

cumar.o: cumar/src/cumar.cc
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o $(OBJECTS_DIR)/cumar.o cumar/src/cumar.cc


cunn: cuda.o cumar.o
	ar cr $(LIB_DIR)/libcunn.a $(OBJECTS_DIR)/cuda.o
	ar cr $(LIB_DIR)/libcunn.a $(OBJECTS_DIR)/cumar.o
	ranlib $(LIB_DIR)/libcunn.a


xor_test: cuda.o cumar.o
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o $(OBJECTS_DIR)/xor_test.o test/xor_test.cc
	$(LINK) -o $(BIN_DIR)/xor_test $(OBJECTS_DIR)/xor_test.o $(OBJECTS_DIR)/cuda.o $(OBJECTS_DIR)/cumar.o $(LFLAGS)


sqrt3: cuda.o cumar.o
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o $(OBJECTS_DIR)/sqrt3.o sqrt3/sqrt3.cc
	$(LINK) -o $(BIN_DIR)/sqrt3 $(OBJECTS_DIR)/sqrt3.o $(OBJECTS_DIR)/cuda.o $(OBJECTS_DIR)/cumar.o $(LFLAGS)

images: cuda.o cumar.o
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o $(OBJECTS_DIR)/images.o images/images.cc
	$(LINK) -o $(BIN_DIR)/images $(OBJECTS_DIR)/images.o $(OBJECTS_DIR)/cuda.o $(OBJECTS_DIR)/cumar.o $(LFLAGS)

mnist: cuda.o cumar.o
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o $(OBJECTS_DIR)/mnist.o mnist/mnist.cc
	$(LINK) -o $(BIN_DIR)/mnist $(OBJECTS_DIR)/mnist.o $(OBJECTS_DIR)/cuda.o $(OBJECTS_DIR)/cumar.o $(LFLAGS)

