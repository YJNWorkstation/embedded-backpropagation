CC := g++
BUILD_DIR := ./build
EXAMPLE_SRC := ./examples

EXAMPLE_BIN := $(patsubst $(EXAMPLE_SRC)/%.cpp,$(BUILD_DIR)/%, $(wildcard $(EXAMPLE_SRC)/*.cpp))

examples: clean | $(EXAMPLE_BIN)

$(BUILD_DIR)/%: $(EXAMPLE_SRC)/%.cpp | dir
	$(CC) -std=c++20 -I ./ -o $@ $<

dir:
	mkdir $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
