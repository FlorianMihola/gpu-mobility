CUDA_PATH ?= /usr/local/cuda-8.0

HOST_COMPILER ?= g++
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

GLFWLIBS := `pkg-config --static --libs glfw3`

CC := gcc

LIBS :=
#LIBS += $(GLFWLIBS)
LIBS += -lm

INCLUDES :=
INCLUDES += -Isrc
#INCLUDES += -Iglad/include

NVCCFLAGS := -Wno-deprecated-gpu-targets  # -m64 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60 
all: build

#build: dist glad dist/mobility
build: dist dist/mobility

dist:
	mkdir dist

glad:
	python -m glad --profile compatibility --generator c --out-path glad

dist/glad.o: glad/src/glad.c
	$(CC) $(INCLUDES) -o $@ -c $<

glad/src/glad.c: glad

dist/mobility: dist/main.o # dist/LinkedList.o # dist/glad.o
	$(NVCC) $(NVCCFLAGS) -o $@ $+ $(LIBS)

dist/main.o: src/main.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ -c $<

dist/LinkedList.o: src/LinkedList.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ -c $<

clean: cleanSoft

cleanSoft:
	rm -fr dist

cleanHard:
	rm -fr dist glad

run: build
	./dist/mobility
