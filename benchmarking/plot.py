import matplotlib.pyplot as plt
import pandas as pd

plt.figure(figsize=(5, 4), layout="constrained")
colors = ["#009AF9", "#E26F46"]
for file, color in zip(
    ["benchmark-cpu.csv", "benchmark-NVIDIA_GeForce_RTX_4090.csv"], colors
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
plt.ylim(top=1, bottom=1e-3)
plt.legend(frameon=True)
plt.savefig("assembly_time.png")
plt.show()
