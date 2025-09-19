import os
import subprocess

os.environ["JAX_PLATFORM"] = "rocm-gpu"


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
#SBATCH --gpus-per-node={gpu_model}:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --job-name=benchmarking
#SBATCH --output={job_folder}/{gpu_model}.out
#SBATCH --error={job_folder}/{gpu_model}.err

module load stack/2024-06 python/3.12.8 eth_proxy 
module load aqlprofile/6.3.2 comgr/6.3.2 composable-kernel/6.3.2 hip/6.3.2 hipblas/6.3.2 hipblas-common/6.3.2 hipblaslt/6.3.2 hipcc/6.3.2
module load hipcub/6.3.2 
module load hipfft/6.3.2
module load hipfort/6.3.2
module load hipify-clang/6.3.2
module load hiprand/6.3.2
module load hipsolver/6.3.2
module load hipsparse/6.3.2
module load hiptt/master
module load hsa-rocr-dev/6.3.2
module load llvm-amdgpu/6.3.2
module load miopen-hip/6.3.2
module load rccl/6.3.2
module load rocblas/6.3.2
module load rocfft/6.3.2
module load rocm-cmake/6.3.2
module load rocm-core/6.3.2
module load rocminfo/6.3.2
module load rocm-openmp-extras/6.3.2
module load rocm-smi-lib/6.3.2
module load rocm-tensile/6.3.2
module load rocmlir/6.3.2
module load rocprim/6.3.2
module load rocprofiler-dev/6.3.2
module load rocprofiler-register/6.3.2
module load rocrand/6.3.2
module load rocsolver/6.3.2
module load rocsparse/6.3.2
module load rocthrust/6.3.2
module load roctracer-dev/6.3.2
module load roctracer-dev-api/6.3.2

source /cluster/home/mpundir/python-venv/rocm632_python/bin/activate
export JAX_CACHE_DIR="$SCRATCH/jax-cache-rocm"
python {file_to_run}

"""
    job_name = f"""job-{gpu_model}.sh"""

    job_script_name = job_folder + "/" + job_name
    open(job_script_name, "w").write(script)

    return job_script_name


# this loop creates the necessay files needed for running simulation for each parametric point
for gpu_model in [
    "mi300a",
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
