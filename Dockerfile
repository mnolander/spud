# Use the official ROS Noetic (Ubuntu 20.04) image as the base
FROM osrf/ros:noetic-desktop-full

# Set environment to noninteractive to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install system dependencies and ROS packages (no Kalibr-related packages)
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    usbutils \
    ros-noetic-usb-cam \
    ros-noetic-rospy \
    ros-noetic-bag \
    build-essential \
    cmake \
    git \
    libeigen3-dev \
    libboost-all-dev \
    libsuitesparse-dev \
    libblas-dev \
    liblapack-dev \
    ros-noetic-vision-opencv \
    ros-noetic-image-transport-plugins \
    ros-noetic-cv-bridge \
    ros-noetic-tf \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    python3-wstool \
    python3-catkin-tools \
    libv4l-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python3 as default
RUN ln -s /usr/bin/python3 /usr/bin/python

# Expose any ports for ROS (optional)
EXPOSE 11311

# Set permissions for access to USB devices
RUN usermod -aG dialout root

# Set the working directory for your Python script
WORKDIR /home/docker

# Copy your Python script into the container (replace 'record_bag.py' with your script)
COPY record_bag.py /home/docker/record_bag.py

# Set up the entrypoint to run the Python script that records a bag file
CMD ["python3", "/home/docker/record_bag.py"]