#!/bin/bash

# Ensure SSH key permissions are correct
chmod 600 /flash/.ssh/id_rsa

# Create .ssh directory in the home directory if it doesn't exist
mkdir -p ~/.ssh

# Symlink the SSH key from /flash to the home directory
ln -sf /flash/.ssh/id_rsa ~/.ssh/id_rsa
ln -sf /flash/.ssh/id_rsa.pub ~/.ssh/id_rsa.pub

# Start the SSH agent and add the SSH key
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa

# Add GitHub to known hosts to avoid prompt
ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts