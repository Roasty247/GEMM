# Makefile for Matrix Multiplication Performance Challenge
# Optimized for Apple Silicon (ARM64)

CC = clang
CFLAGS = -Wall -Wextra
SRC_DIR = src
BUILD_DIR = build
RESULTS_DIR = results
MAIN = $(SRC_DIR)/main.c
TARGETS = matmult matmult_ijp matmult_jpi matmult_pji matmult_blocked

# Debug build for all variants
debug: CFLAGS += -g -O0 -fsanitize=address
debug: LDFLAGS += -fsanitize=address
debug: $(addprefix $(BUILD_DIR)/, $(addsuffix -debug, $(TARGETS)))

# Optimized build for all variants (We are using -mcpu=apple-m1 Use the right "m" (m1, m2, m3, etc.) version of the CPU)
opt: CFLAGS += -O3 -mcpu=apple-m3 -ffast-math -funroll-loops
opt: $(addprefix $(BUILD_DIR)/, $(addsuffix -opt, $(TARGETS)))

# Individual variant targets
$(BUILD_DIR)/%-debug: $(MAIN) $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

$(BUILD_DIR)/%-opt: $(MAIN) $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $^ -o $@

$(BUILD_DIR) $(RESULTS_DIR):
	mkdir -p $@

# Performance test for all variants
# Binaries run from inside build/ so the variant name in the CSV (argv[0]) stays ./<variant>-opt
perf: opt | $(RESULTS_DIR)
	@first_variant=$$(echo $(TARGETS) | cut -d' ' -f1); \
	(cd $(BUILD_DIR) && ./$$first_variant-opt 100 100 100 ../$(RESULTS_DIR)/perf_$$first_variant.csv); \
	head -n 1 $(RESULTS_DIR)/perf_$$first_variant.csv > $(RESULTS_DIR)/perf.csv; \
	for variant in $(TARGETS); do \
		for size in 100 500 1000 1500 2000; do \
			echo "Running $$variant with size $$size..."; \
			(cd $(BUILD_DIR) && ./$$variant-opt $$size $$size $$size ../$(RESULTS_DIR)/perf_$$variant.csv); \
			tail -n 1 $(RESULTS_DIR)/perf_$$variant.csv >> $(RESULTS_DIR)/perf.csv; \
		done; \
	done
	@rm -f $(RESULTS_DIR)/perf_*.csv

# Test individual variant
test-%: $(BUILD_DIR)/%-opt | $(RESULTS_DIR)
	@echo "Testing $*..."
	@./$(BUILD_DIR)/$*-opt 1000 1000 1000 $(RESULTS_DIR)/test_$*.csv
	@cat $(RESULTS_DIR)/test_$*.csv
	@rm -f $(RESULTS_DIR)/test_$*.csv

# Clean
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(RESULTS_DIR)/perf.csv $(RESULTS_DIR)/test_*.csv

.PHONY: debug opt perf clean test-%
