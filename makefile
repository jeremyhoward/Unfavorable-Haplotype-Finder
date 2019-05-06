#-----------------------------------#
# Make file for ROH-Haplo Finder#
#
# Supported platforms: Unix/Linux
#-----------------------------------#

# Director of the target
OUTPUT = Haplofinder

# Compiler
CXX = g++

# mkl and eigen paths (change if compiling on own)
MKLPATH = /home/jthoward/mkl/mkl
EIGEN_PATH = /home/jthoward/Simulation/Ne1000/eigen-eigen-bdd17ee3b1b3

# Compiler flags
CXXFLAGS = -O3 -DMKL_ILP64 -m64 -fopenmp -std=c++11 -I ${EIGEN_PATH} -I ${MKLPATH}/include
LIBS = -L$(MKLPATH)/lib/intel64
LDFLAGS =  -Wl,--start-group ${MKLPATH}/lib/intel64/libmkl_intel_ilp64.a ${MKLPATH}/lib/intel64/libmkl_core.a ${MKLPATH}/lib/intel64/libmkl_gnu_thread.a -Wl,--end-group -lpthread -lm -ldl

HDR += HaplofinderClasses.h

SRC = HaplofinderClasses.cpp \
	Scan_Genome_Final.cpp

OBJ = $(SRC:.cpp=.o)

all : $(OUTPUT)

$(OUTPUT) :
	$(CXX) $(CXXFLAGS) -o $(OUTPUT) $(OBJ) $(LIB) $(LDFLAGS)

$(OBJ) : $(HDR)

.cpp.o :
	$(CXX) $(CXXFLAGS) -c $*.cpp
.SUFFIXES : .cpp .c .o $(SUFFIXES)

$(OUTPUT) : $(OBJ)

FORCE:

clean:
	rm -f *.o
