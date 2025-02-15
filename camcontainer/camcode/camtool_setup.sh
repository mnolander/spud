#!/bin/bash

set -e  # Exit on error
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y \
    python3 \
    python3-opencv \
    python3-pip \
    curl \
    git \
    wget

rm -rf /var/lib/apt/lists/*

# Install ROS

apt-get install -y software-properties-common
add-apt-repository universe
add-apt-repository multiverse
add-apt-repository restricted

apt-get update
sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
curl -sSL 'https://raw.githubusercontent.com/ros/rosdistro/master/ros.key' | sudo tee /usr/share/keyrings/ros-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" | sudo tee /etc/apt/sources.list.d/ros-latest.list > /dev/null


curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

apt-get update

apt-get install -y ros-noetic-desktop-full

apt-get update
# Ros packages
apt-get install -y \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    ros-noetic-ros-base=1.5.0-1* \
    ros-noetic-mavros \
    ros-noetic-mavros-extras \
    ros-noetic-usb-cam \
    

rm -rf /var/lib/apt/lists/*

wget https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh
chmod +x install_geographiclib_datasets.sh
./install_geographiclib_datasets.sh
rm install_geographiclib_datasets.sh

# cloning repositories
cd ~
git clone https://github.com/admiraldre/dronecamtoolbox.git
git clone https://github.com/hoangthien94/vision_to_mavros.git


# Install VimbaX

TAR_FILE="VimbaX_Setup-2023-4-Linux64.tar.gz" # Change to whichever version you have
INSTALL_DIR="/opt/VimbaX"

# Ensure the tar file exists
if [[ ! -f "$TAR_FILE" ]]; then
    echo "Error: File '$TAR_FILE' not found!"
    exit 1
fi

echo "Extracting $TAR_FILE..."
mkdir -p "$INSTALL_DIR"
tar -xzf "$TAR_FILE" -C "$INSTALL_DIR" --strip-components=1

# Set environment variables
echo "Setting up environment variables..."
echo 'export VIMBAX_HOME="/opt/VimbaX"' | tee /etc/profile.d/vimbax.sh
echo 'export PATH="$VIMBAX_HOME:$PATH"' | tee -a /etc/profile.d/vimbax.sh
chmod +x /etc/profile.d/vimbax.sh
echo 'source /etc/profile.d/vimbax.sh'