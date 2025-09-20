import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd
import numpy as np
# "benchmark-NVIDIA_GeForce_RTX_4090.csv"

plt.figure(figsize=(5, 4), layout="constrained")
colors = ["#009AF9", "#E26F46", "#3DA44D", "#FF6F61", '#922B21']

files = [
        "benchmark-vmap_cpu.csv",
        "benchmark-chunked_vmap_cpu.csv",
        "benchmark-NVIDIA_GeForce_RTX_4090.csv",
        "benchmark-chunked_vmap_mi300a.csv",
        "benchmark-vmap_mi300a.csv",
    ]

colors = cm.rainbow(np.linspace(0, 1, len(files)))

for file, color in zip(
    files,
    colors,
):
    results = pd.read_csv(file)
    plt.loglog(
        results["size"],
        results["stiffness_exe_time"],
        "o-",
        color=color,
        label=file.split("-")[1].split(".")[0],
    )

plt.xlabel("Number of DOFs")
plt.ylabel("Time (s)")
plt.title("Assembly Time")
plt.grid(True)
plt.ylim(bottom=1e-3)
plt.legend(frameon=True)
plt.savefig("assembly_time.png")
plt.show()
