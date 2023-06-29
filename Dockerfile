FROM nvidia/cuda:11.7.2-devel-ubuntu20.04
ENV PYTHONUNBUFFERED 1

# Setting up basic repo 
ARG DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Setting up working directory
COPY . project/
WORKDIR /project
ENV RUNNING_IN_DOCKER True
ENV CONDA_DIR "/project/conda"
ENV VENV_DIR "/project/venv"

# Install prerequisits
RUN apt add-repository -y ppa:deadsnakes/ppa apt-get update && apt-get install -y apt-utils \
    software-properties-common \
    make build-essential wget curl git nano ffmpeg libsm6 libxext6 \
    p7zip-full p7zip-rar \
    python3.10-full python-is-python3 \
    python3-notebook jupyter jupyter-core \ 
    pkg-config libcairo2-dev libjpeg-dev libgif-dev && apt-get clean -y


# Download and install miniconda
RUN curl -Lk "https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh" > "miniconda_installer.sh" \
    && chmod u+x "miniconda_installer.sh" \
    && /bin/bash "miniconda_installer.sh" -b -p "$CONDA_DIR" \
    && echo "Installed miniconda version:" \
    && "${CONDA_DIR}/bin/conda" --version

# Create conda environment
RUN "${CONDA_DIR}/bin/conda" create -y -k --prefix "$VENV_DIR" python=3.10

# Networking
ENV PORT 7860
EXPOSE $PORT

# Setting up conda environment
RUN /bin/bash /project/install.sh

# Install as kernel
RUN ipython kernel install --user --name=venv

# Start text-generation-webui
CMD ["/bin/bash", "run.sh"]
