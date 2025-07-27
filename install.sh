#!/bin/bash
echo "The following commands require sudo privileges. Please enter your password if prompted."
sudo apt-get update

# pip packages install
pip install -r requirements.txt

# qrcode install
pip install "qrcode[pil]"

# System package (sudo privileges)
sudo apt-get install -y libzbar0
sudo apt-get install -y libdmtx0b

# Python packages
pip install pyzbar
pip install pylibdmtx
