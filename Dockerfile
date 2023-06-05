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
ENV VENV_DIR "/project/venv"

# Install prerequisits
RUN apt-get update && apt-get install -y apt-utils \
    software-properties-common \
    make build-essential wget curl git nano ffmpeg libsm6 libxext6 \
    p7zip-full p7zip-rar \
    python3-pip python3-venv \
    pkg-config libcairo2-dev libjpeg-dev libgif-dev && apt-get clean -y


# Create venv
RUN if [ ! -d "venv" ]; \
    then \
    python3 -m venv venv; \
    fi 

# Networking
ENV PORT 7860
EXPOSE $PORT

# Setting up text-generation-webui
RUN source /project/venv && python -m pip install -r /project/requirements.txt

# Start text-generation-webui
CMD ["/bin/bash", "run.sh"]