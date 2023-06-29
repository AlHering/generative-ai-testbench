#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

printf "\n%s\n" "${delimiter}"
printf "Activating conda environment"
printf "\n%s\n" "${delimiter}"
 
if [[ -f "${CONDA_DIR}/etc/profile.d/conda.sh"  ]]
then
    source "${CONDA_DIR}/etc/profile.d/conda.sh"
else
    printf "\n%s\n" "${delimiter}"
    printf "\e[1m\e[31mERROR: Cannot initialize conda, aborting...\e[0m"
    printf "\n%s\n" "${delimiter}"
    exit 1
fi 

printf "\n%s\n" "${delimiter}"
printf "Setting up conda environment..."
printf "\n%s\n" "${delimiter}"
conda activate "$VENV_DIR"
conda install -y -k pytorch[version=2,build=py3.10_cuda11.7*] torchvision torchaudio pytorch-cuda=11.7 cuda-toolkit ninja git -c pytorch -c nvidia/label/cuda-11.7.0 -c nvidia

printf "\n%s\n" "${delimiter}"
printf "Handling main webui requirements..."
printf "\n%s\n" "${delimiter}"

python -m pip install -r requirements.txt

cd "${SCRIPT_DIR}"

printf "\n%s\n" "${delimiter}"
printf "Finished installation"
printf "\n%s\n" "${delimiter}"
exit 0