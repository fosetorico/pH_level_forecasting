# pH Level Forecasting of Well Water Samples in Malawi
This project focuses on an End-to-End Project for Forecasting pH levels of water samples in Malawi, which was conducted by Leeds Beckett University.

## Workflows
- Understanding the Problem Statement
- Exploratory data analysis
- Data Pre-Processing
- Model Training
- Pipelining
    - Data Ingestion
    - Data Transformation
    - Model Trainer
    - Prediction Pipelining using Flask

## Running Instruction

Clone the Repository

```bash
https://github.com/fosetorico/pH_level_forecasting.git
```

Create a conda environment after opening the repository

```bash
conda create -n your-chosen-name python=3.10 -y
```

```bash
conda activate your-chosen-name
```

## Install the requirements

```bash
pip install -r requirements.txt
```

## Finally run the app
```bash
python app.py
```

### Now,
```bash
open up you local host and port
```

## AWS-CICD-Deployment-with-Github-Actions

	With specific access:

	1. EC2 access : It is virtual machine
	2. ECR: Elastic Container registry to save your docker image in aws

	Description: About the deployment

	1. Build docker image of the source code
	2. Push your docker image to ECR
	3. Launch Your EC2 
	4. Pull Your image from ECR in EC2
	5. Lauch your docker image in EC2

	Policy:

	1. AmazonEC2ContainerRegistryFullAccess
	2. AmazonEC2FullAccess

### 1. Login to AWS console.

### 2. Create IAM user for deployment

### 3. Create ECR repo to store/save docker image

### 4. Create EC2 machine (Ubuntu) 

### 5. Open EC2 and Install docker in EC2 Machine:

### 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one

	>>>Optinal<<<

	sudo apt-get update -y

	sudo apt-get upgrade
	
	>>>Required<<<

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker

# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID

    AWS_SECRET_ACCESS_KEY

    AWS_REGION

    AWS_ECR_LOGIN_URI

    ECR_REPOSITORY_NAME