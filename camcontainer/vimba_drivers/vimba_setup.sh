#!/bin/bash

set -e  # Exit on error

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