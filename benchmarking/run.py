import os
import subprocess


# file to execute
job_folder = "/cluster/home/mpundir/dev/femsolver/benchmarking"

runfile = "/cluster/home/mpundir/dev/femsolver/benchmarking/profiling_gpus.py"


def create_sbatch_script(file_to_run, job_folder, gpu_model):
    """function for specifying options for a serial run

    Args:
        file_to_run (str): path to the executable file
        input_file (str): path to the input file with all the parameters
        idx (str): unique id for a job or for a parametric point
        job_folder (str): path to the folder where results for the job will be stored
        nb_nodes (int, optional): _description_. Defaults to 1.
    """
    script = f"""#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --gpus-per-node={gpu_model}:2
#SBATCH --gres=gpumem:24g
#SBATCH --time=01:00:00
#SBATCH --job-name=benchmarking
#SBATCH --output={job_folder}/{gpu_model}.out
#SBATCH --error={job_folder}/{gpu_model}.err

source /cluster/project/cmbm/local-stacks/load-scripts/load_gpu.sh 
source /cluster/home/mpundir/python-venv/test/bin/activate
export JAX_CACHE_DIR="$SCRATCH/jax-cache-gpu"
export JAX_PLATFORM="gpu"
python {file_to_run}

"""
    job_name = f"""job-{gpu_model}.sh"""

    job_script_name = job_folder + "/" + job_name
    open(job_script_name, "w").write(script)

    return job_script_name


# this loop creates the necessay files needed for running simulation for each parametric point
for gpu_model in [
    "rtx_4090",
    #"v100",
    #"rtx_3090",
    #"titan_rtx",
    #"quadro_rtx_6000",
    #"a100-pcie-40gb",
]:

    job_script_name = create_sbatch_script(
        file_to_run=runfile,
        job_folder=job_folder,
        gpu_model=gpu_model,
    )

    result = subprocess.run(
        "sbatch --parsable {}".format(job_script_name),
        capture_output=True,
        text=True,
        shell=True,
        env=os.environ,
    ).stdout.strip("\n")

    print("Job | ", result)
