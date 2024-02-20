# For this specific script we will use python 3.11.5
FROM --platform=linux/amd64 python:3.11.5 as build

# This is our working directory in the container
WORKDIR /app

# Copy the required files
COPY ai/dog_class.py .
COPY ai/dog.jpg .
COPY ai/imagenet_classes.txt .

# It's important to install these specific versions 
RUN pip install torch==2.0.0 torchvision==0.15.1

# Command to run the Python script
CMD ["python", "dog_class.py"]
