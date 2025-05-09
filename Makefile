CC = g++
CFLAGS = -std=c++17 -O3 \
		-DFALCONN_NO_FHT \
        -I./FALCONN/src/include \
        -I./FALCONN/external/eigen \
        -I./FALCONN/external/simple-serializer \
		-Wno-deprecated-declarations

OBJS = ANN.o datatype.o quad_tree.o balanced_binary_tree.o LSH.o main.o misc.o random.o aspect_ratio_depth.o BruteForceNN.o data_preprocess.o baseline/benchmark.o ANN-Oracle.o point.o

all: text_embedding

text_embedding: $(OBJS) text_embedding.o
	@if [ -d text_embedding ]; then \
		echo "⚠️  Directory named 'text_embedding' exists. Removing to avoid conflict..."; \
		rm -rf text_embedding; \
	fi
	$(CC) $(CFLAGS) $(OBJS) text_embedding.o -o text_embedding

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf *.o *.out text_embedding
