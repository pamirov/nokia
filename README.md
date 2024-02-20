# Nokia

This is a temporary repo for the **nokia** assignment.

## Overview

In this mini project I wrote a Python script that perfoms image classification using a pre-trained deep learning model, the script is called `dog_class.py`, there you can also find a picture that must be classified which is called `dog.jpg`.

I have integrated this script into a CI/CD pipeline using GitHub Actions with 3 workflows, those are:
- Unit test & app validation
- Docker image build & push
- Deploy to EKS

So the first workflow runs the unit test using `pytest` against the code and then run the code itself and shows the result.
The second workflow containerize the Python app by building a Docker image and then pushes it to my public Dockerhub.
The third workflow deploys the containerized app to AWS EKS cluster.

## Validation

Finally to validate that everything works as expected, we can just check the logs of the pod, and it should show the result of the image classification, the image is a little puppy, and the correct result is "curly-coated retriever".
