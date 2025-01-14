# Use the official ROS Noetic (Ubuntu 20.04) image as the base
FROM ros:noetic-ros-core-focal

# Set environment to noninteractive to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install system dependencies, including Kalibr dependencies, Xvfb, and ROS packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    usbutils \
    ros-noetic-usb-cam \
    xvfb \
    ros-noetic-rospy \
    x11-apps \
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

# Set environment variable for GENICAM
ENV GENICAM_GENTL64_PATH=/home/docker/camcontainer/Vimba_2024-1/cti
ENV VIMBAX_HOME /home/docker/camcontainer/VimbaX_2024-1
ENV LD_LIBRARY_PATH $VIMBAX_HOME/api/lib:$LD_LIBRARY_PATH

# Copy the local "camcontainer" directory into the Docker image - Replace "camcontainer" with the name of your directory
COPY camcontainer /home/docker/camcontainer

# # Set the working directory
WORKDIR /home/docker/camcontainer


# Clone and set up Kalibr
# RUN git clone https://github.com/ethz-asl/kalibr.git /home/docker/camcontainer/kalibr && \
#     mkdir -p /home/docker/camcontainer/kalibr_workspace/src && \
#     ln -s /home/docker/camcontainer/kalibr /home/docker/camcontainer/kalibr_workspace/src/kalibr && \
#     cd /home/docker/camcontainer/kalibr_workspace && \
#     /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin init && catkin build"

# Add Kalibr environment setup to bashrc
# RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
#     echo "source /home/docker/camcontainer/kalibr_workspace/devel/setup.bash" >> ~/.bashrc

# Expose any ports, if necessary (e.g., for ROS nodes)
EXPOSE 11311

# Set permissions for access to USB devices
RUN usermod -aG dialout root

# Set up the entrypoint to run a bash shell
CMD ["/bin/bash"]
