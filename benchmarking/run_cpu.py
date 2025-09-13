import os
import subprocess

os.environ["JAX_PLATFORM"] = "cpu"

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
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8096
#SBATCH --time=01:00:00
#SBATCH --job-name=benchmarking
#SBATCH --output={job_folder}/{gpu_model}.out
#SBATCH --error={job_folder}/{gpu_model}.err

source /cluster/project/cmbm/local-stacks/load-scripts/load_fenicsx.sh 
source /cluster/home/mpundir/python-venv/cpu_python/bin/activate
export JAX_CACHE_DIR="$SCRATCH/jax-cache-cpu"
python {file_to_run}

"""
    job_name = f"""job-cpu.sh"""

    job_script_name = job_folder + "/" + job_name
    open(job_script_name, "w").write(script)

    return job_script_name


job_script_name = create_sbatch_script(
     file_to_run=runfile,
     job_folder=job_folder,
     gpu_model="cpu",
 )
result = subprocess.run(
     "sbatch --parsable {}".format(job_script_name),
     capture_output=True,
     text=True,
     shell=True,
     env=os.environ,
 ).stdout.strip("\n")
print("Job | ", result)
