#!/bin/bash

set -e  # Exit on error

# Add ROS package repository
apt-get update && apt-get install -y software-properties-common
add-apt-repository universe
add-apt-repository multiverse
add-apt-repository restricted

# Add ROS Noetic package source
sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

# Add ROS key
apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 #change key

# Update package list after adding ROS repo
apt-get update

# Install dependencies
apt-get update && apt-get install -y \
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
    libv4l-dev

TAR_FILE="VimbaX_Setup-2023-4-Linux64.tar.gz"  # Change to the actual version if needed
INSTALL_DIR="/opt/VimbaX"
PROFILE_SCRIPT="/etc/profile.d/vimbax.sh"

# Ensure the tar file exists
if [[ ! -f "$TAR_FILE" ]]; then
    echo "Error: File '$TAR_FILE' not found!"
    exit 1
fi

# Create the installation directory if it doesn't exist
mkdir -p "$INSTALL_DIR"

# Extract the tar file
echo "Extracting $TAR_FILE to $INSTALL_DIR..."
tar -xzf "$TAR_FILE" -C "$INSTALL_DIR" --strip-components=1

# Set up environment variables
cat <<EOF | tee "$PROFILE_SCRIPT"
export VIMBAX_HOME="$INSTALL_DIR"
export PATH="\$VIMBAX_HOME:\$PATH"
EOF

# Ensure the profile script is executable
chmod +x "$PROFILE_SCRIPT"

# Source the profile script immediately for the current session
echo "Applying environment variables..."
source "$PROFILE_SCRIPT"

echo "VimbaX installation completed successfully."