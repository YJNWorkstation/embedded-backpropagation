CC := g++
SRC := ./src
BUILD_DIR := ./build
EXAMPLE_SRC := ./examples
LIB_NAME := backpropagation

SRCS := $(wildcard $(SRC)/*.cpp)
OBJS    := $(patsubst $(SRC)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))

linear_graph: lib
	$(CC) -o $(BUILD_DIR)/linear_graph $(EXAMPLE_SRC)/linear_graph.cpp -L$(BUILD_DIR) -l:$(LIB_NAME).a

lib: clean | $(OBJS)
	ar csrf $(BUILD_DIR)/$(LIB_NAME).a $(OBJS)

$(BUILD_DIR)/%.o: $(SRC)/%.cpp | dir
	$(CC) -c $< -o $@

dir:
	mkdir $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
