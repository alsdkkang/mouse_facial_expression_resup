# Server Environment Setup Guide

This guide explains how to set up the Python environment on your lab server (`abizaidlab-ml`) using the provided script.

## Prerequisites
- You must be connected to the server via VS Code (see [SERVER_CONNECTION_GUIDE.md](SERVER_CONNECTION_GUIDE.md)).
- The project files must be on the server (VS Code usually handles this if you open the folder on the server).

## Step 1: Open Terminal in VS Code (Server)
1.  Connect to the server using VS Code.
2.  Open the terminal (`Ctrl + ~` or Terminal > New Terminal).
3.  Ensure you are in the project root directory.

## Step 2: Run the Setup Script
Run the following commands to make the script executable and execute it:

```bash
chmod +x setup_server_env.sh
./setup_server_env.sh
```

This script will:
1.  Create a new conda environment named `mouse_face`.
2.  Install PyTorch with CUDA acceleration.
3.  Install all required libraries (pandas, matplotlib, lightning, etc.).
4.  Install the current project in editable mode.

## Step 3: Verify Installation
After the script finishes, verify that the environment works and sees the GPU:

1.  Activate the environment:
    ```bash
    conda activate mouse_face
    ```
2.  Run this quick check:
    ```bash
    python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Count: {torch.cuda.device_count()}')"
    ```
    - **Expected Output**: `CUDA Available: True`, `Device Count: 1` (or more).

## Troubleshooting
- **Conda not found**: If it says `conda: command not found`, try restarting your terminal or running `source ~/.bashrc`.
- **Permission denied**: Make sure you ran `chmod +x setup_server_env.sh`.
