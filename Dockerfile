# Step 1: Use an official Python runtime as a parent image
# Specify the platform to ensure compatibility with the amd64 architecture
FROM --platform=linux/amd64 python:3.9-slim

# Step 2: Set the working directory inside the container
# All subsequent commands will be run from this directory
WORKDIR /app

# Step 3: Copy the requirements file into the container
# This is done first to leverage Docker's layer caching. The next step will only
# re-run if the requirements.txt file changes.
COPY requirements.txt .

# Step 4: Install the Python dependencies
# The --no-cache-dir option is used to keep the final image size smaller.
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy your application files into the container
# This includes the inference script and the trained model file.
COPY run_inference.py .
COPY document_structure_model.joblib .

# Step 6: Specify the command to run when the container starts
# This command executes your inference script. The script is already designed
# to read from /app/input and write to /app/output, which will be mapped
# by the `docker run` command's volume mounts.
CMD ["python", "run_inference.py"]