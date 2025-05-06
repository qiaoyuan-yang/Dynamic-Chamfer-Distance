CC = g++
CFLAGS = -std=c++17 -O0 -g \
		-DFALCONN_NO_FHT \
        -I./FALCONN/src/include \
        -I./FALCONN/external/eigen \
        -I./FALCONN/external/simple-serializer \
		-Wno-deprecated-declarations

OBJS = ANN.o datatype.o quad_tree.o balanced_binary_tree.o LSH.o main.o misc.o random.o aspect_ratio_depth.o BruteForceANN.o data_preprocess.o baseline/benchmark.o

all: data

data: $(OBJS) data.o
	@if [ -d data ]; then \
		echo "⚠️  Directory named 'data' exists. Removing to avoid conflict..."; \
		rm -rf data; \
	fi
	$(CC) $(CFLAGS) $(OBJS) data.o -o data

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf *.o *.out data
