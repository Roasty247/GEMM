import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('results/perf.csv', sep=',')

df.columns = df.columns.str.strip()

# === First plot: Execution time ===
plt.figure(figsize=(10, 6))
plt.yscale('log')

for variant, data in df.groupby('variant'):
    plt.plot(data['M'], data['seconds'], marker='o', label=variant)

plt.title("Execution time Comparison of Matrix Multiplication Variants")
plt.xlabel("Matrix Dimension (M)")
plt.ylabel("Execution Time (seconds)")
plt.legend(title="Variant")
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('results/matmult_performance.png', dpi=300)
plt.show()


# === Second plot: Throughput ===
plt.figure(figsize=(10, 6))

for variant, data in df.groupby('variant'):
    plt.plot(data['M'], data['throughput'], marker='o', label=variant)

plt.title("Throughput Comparison of Matrix Multiplication Variants")
plt.xlabel("Matrix Dimension (M)")
plt.ylabel("Throughput (GFLOP/s)")
plt.legend(title="Variant")
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('results/matmult_throughput.png', dpi=300)
plt.show()

