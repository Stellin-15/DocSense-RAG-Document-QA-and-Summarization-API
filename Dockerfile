# File: Dockerfile

# Stage 1: Use an official Python runtime as a parent image
# Using a slim-buster image keeps the final image size smaller.
FROM python:3.11-slim-buster

# Stage 2: Set up the working environment
WORKDIR /app

# Prevent Python from writing .pyc files to disc
ENV PYTHONDONTWRITEBYTECODE 1
# Ensure Python output is sent straight to the terminal without buffering
ENV PYTHONUNBUFFERED 1

# Stage 3: Install dependencies
# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed system-level dependencies (if any)
# RUN apt-get update && apt-get install -y ...

# Install the Python dependencies specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Stage 4: Copy application code
# Copy the rest of the application's code into the container at /app
COPY . .

# Stage 5: Expose the port the app runs on
# FastAPI/Uvicorn will run on port 8000 by default
EXPOSE 8000

# Stage 6: Define the command to run the application
# This command starts the Uvicorn server.
# --host 0.0.0.0 makes the server accessible from outside the container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]