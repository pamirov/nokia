# Use the official Python image as the base image
FROM --platform=linux/amd64 python:3.11.5 as build

# Set the working directory in the container
WORKDIR /app

# Copy the Python script and required files into the container
COPY ai/dog_class.py .
COPY ai/dog.jpg .
COPY ai/imagenet_classes.txt .

# Install specific versions of torch and torchvision
RUN pip install torch==2.0.0 torchvision==0.15.1

# Command to run the Python script
CMD ["python", "dog_class.py"]
