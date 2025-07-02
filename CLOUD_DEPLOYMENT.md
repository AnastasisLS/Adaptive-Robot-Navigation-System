# AWS Cloud Deployment Guide for Adaptive Robot Navigation System

## Overview
This guide will help you deploy your experiments to AWS EC2, allowing you to run long experiments without keeping your computer on. We'll use AWS EC2 with auto-scaling and S3 for result storage.

## Prerequisites
1. AWS Account (free tier available)
2. AWS CLI installed locally
3. Basic knowledge of AWS services

## Step 1: AWS Setup

### Install AWS CLI
```bash
# macOS
brew install awscli

# Or download from AWS website
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /
```

### Configure AWS CLI
```bash
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Enter your default region (e.g., us-east-1)
# Enter your output format (json)
```

## Step 2: Create AWS Infrastructure

### Create S3 Bucket for Results
```bash
# Create bucket (replace with unique name)
aws s3 mb s3://adaptive-robot-experiments-$(date +%s)

# Set bucket policy for public read access to results
aws s3api put-bucket-policy --bucket adaptive-robot-experiments-$(date +%s) --policy '{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PublicReadGetObject",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::adaptive-robot-experiments-$(date +%s)/results/*"
        }
    ]
}'
```

### Create IAM Role for EC2
```bash
# Create trust policy
cat > trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create role
aws iam create-role --role-name ExperimentRunnerRole --assume-role-policy-document file://trust-policy.json

# Attach policies
aws iam attach-role-policy --role-name ExperimentRunnerRole --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam attach-role-policy --role-name ExperimentRunnerRole --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess

# Create instance profile
aws iam create-instance-profile --instance-profile-name ExperimentRunnerProfile
aws iam add-role-to-instance-profile --instance-profile-name ExperimentRunnerProfile --role-name ExperimentRunnerRole
```

## Step 3: Prepare Your Project for AWS

### Create Project Archive
```bash
# Create a zip file of your project (excluding unnecessary files)
zip -r project.zip . -x "*.git*" "*.pyc" "__pycache__/*" "*.DS_Store" "data/experiments/*" ".pytest_cache/*"
```

### Upload to S3
```bash
# Upload project to S3
aws s3 cp project.zip s3://adaptive-robot-experiments-$(date +%s)/project.zip
```

## Step 4: Launch EC2 Instance

### Create Launch Template
```bash
# Get the latest Ubuntu AMI
AMI_ID=$(aws ec2 describe-images \
    --owners 099720109477 \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-20.04-amd64-server-*" \
    --query "sort_by(Images, &CreationDate)[-1].ImageId" \
    --output text)

# Create user data script
cat > user-data.sh << 'EOF'
#!/bin/bash
yum update -y
yum install -y python3 python3-pip git

# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install

# Get project from S3
aws s3 cp s3://BUCKET_NAME/project.zip /home/ec2-user/
cd /home/ec2-user
unzip project.zip

# Install dependencies
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install numpy scipy matplotlib seaborn pyyaml gym stable-baselines3 opencv-python tqdm pandas scikit-learn pytest

# Install project
cd "Project 1 - Adaptive Robot Navigation System"
pip3 install -e .

# Run experiments
python3 cloud_setup/headless_experiment.py --experiment basic --episodes 500

# Upload results to S3
aws s3 cp data/experiments/ s3://BUCKET_NAME/results/ --recursive

# Shutdown instance when done
shutdown -h now
EOF

# Create launch template
aws ec2 create-launch-template \
    --launch-template-name ExperimentRunnerTemplate \
    --version-description v1 \
    --launch-template-data "ImageId=$AMI_ID,InstanceType=t3.medium,IamInstanceProfile={Name=ExperimentRunnerProfile},UserData=$(base64 -w 0 user-data.sh)"
```

### Launch Instance
```bash
# Launch instance
aws ec2 run-instances \
    --launch-template LaunchTemplateName=ExperimentRunnerTemplate,Version=1 \
    --instance-type t3.medium \
    --key-name your-key-pair-name \
    --security-group-ids sg-xxxxxxxxx \
    --subnet-id subnet-xxxxxxxxx \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ExperimentRunner}]'
```

## Step 5: Monitor and Retrieve Results

### Monitor Instance
```bash
# Check instance status
aws ec2 describe-instances --filters "Name=tag:Name,Values=ExperimentRunner" --query "Reservations[].Instances[].{InstanceId:InstanceId,State:State.Name,PublicIP:PublicIpAddress}"

# Get instance logs
aws logs describe-log-groups --log-group-name-prefix /aws/ec2/ExperimentRunner
```

### Retrieve Results
```bash
# Download results from S3
aws s3 sync s3://adaptive-robot-experiments-$(date +%s)/results/ ./results/
```

## Step 6: Advanced Setup with Auto Scaling

### Create Auto Scaling Group
```bash
# Create launch template for auto scaling
aws autoscaling create-auto-scaling-group \
    --auto-scaling-group-name ExperimentRunnerASG \
    --launch-template LaunchTemplateName=ExperimentRunnerTemplate,Version=1 \
    --min-size 0 \
    --max-size 5 \
    --desired-capacity 1 \
    --vpc-zone-identifier "subnet-xxxxxxxxx,subnet-yyyyyyyyy"
```

### Create CloudWatch Dashboard
```bash
# Create dashboard for monitoring
aws cloudwatch put-dashboard \
    --dashboard-name ExperimentRunnerDashboard \
    --dashboard-body '{
        "widgets": [
            {
                "type": "metric",
                "properties": {
                    "metrics": [
                        ["AWS/EC2", "CPUUtilization", "AutoScalingGroupName", "ExperimentRunnerASG"]
                    ],
                    "period": 300,
                    "stat": "Average",
                    "region": "us-east-1",
                    "title": "CPU Utilization"
                }
            }
        ]
    }'
```

## Step 7: Cost Optimization

### Use Spot Instances
```bash
# Create spot fleet request for cost savings
aws ec2 request-spot-fleet \
    --spot-fleet-request-config '{
        "AllocationStrategy": "lowestPrice",
        "TargetCapacity": 1,
        "SpotPrice": "0.05",
        "LaunchSpecifications": [
            {
                "ImageId": "'$AMI_ID'",
                "InstanceType": "t3.medium",
                "SubnetId": "subnet-xxxxxxxxx",
                "IamInstanceProfile": {"Name": "ExperimentRunnerProfile"},
                "UserData": "'$(base64 -w 0 user-data.sh)'"
            }
        ]
    }'
```

### Set Up Budget Alerts
```bash
# Create budget alert
aws budgets create-budget \
    --account-id $(aws sts get-caller-identity --query Account --output text) \
    --budget '{
        "BudgetName": "ExperimentBudget",
        "BudgetLimit": {
            "Amount": "10.00",
            "Unit": "USD"
        },
        "TimeUnit": "MONTHLY",
        "BudgetType": "COST"
    }' \
    --notifications-with-subscribers '[
        {
            "Notification": {
                "ComparisonOperator": "GREATER_THAN",
                "NotificationType": "ACTUAL",
                "Threshold": 80.0,
                "ThresholdType": "PERCENTAGE"
            },
            "Subscribers": [
                {
                    "Address": "your-email@example.com",
                    "SubscriptionType": "EMAIL"
                }
            ]
        }
    ]'
```

## Step 8: Automation Scripts

### Create Deployment Script
```bash
cat > deploy_experiment.sh << 'EOF'
#!/bin/bash

# Configuration
BUCKET_NAME="adaptive-robot-experiments-$(date +%s)"
REGION="us-east-1"
INSTANCE_TYPE="t3.medium"
EXPERIMENT_TYPE=${1:-basic}
EPISODES=${2:-500}

echo "Deploying experiment: $EXPERIMENT_TYPE with $EPISODES episodes"

# Create S3 bucket
aws s3 mb s3://$BUCKET_NAME --region $REGION

# Upload project
aws s3 cp project.zip s3://$BUCKET_NAME/project.zip

# Update user data with experiment parameters
sed "s/BUCKET_NAME/$BUCKET_NAME/g" user-data.sh > user-data-updated.sh
sed -i "s/basic/$EXPERIMENT_TYPE/g" user-data-updated.sh
sed -i "s/500/$EPISODES/g" user-data-updated.sh

# Launch instance
INSTANCE_ID=$(aws ec2 run-instances \
    --launch-template LaunchTemplateName=ExperimentRunnerTemplate,Version=1 \
    --instance-type $INSTANCE_TYPE \
    --user-data file://user-data-updated.sh \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instance launched: $INSTANCE_ID"
echo "Bucket created: $BUCKET_NAME"
echo "Monitor progress: aws s3 ls s3://$BUCKET_NAME/results/"
EOF

chmod +x deploy_experiment.sh
```

## Usage Examples

### Run Basic Experiment
```bash
./deploy_experiment.sh basic 500
```

### Run Comparison Experiment
```bash
./deploy_experiment.sh comparison 100
```

### Monitor Progress
```bash
# Check if results are being generated
aws s3 ls s3://your-bucket-name/results/

# Download results when complete
aws s3 sync s3://your-bucket-name/results/ ./local-results/
```

## Cost Estimation

### T3.Medium Instance (Recommended)
- **CPU**: 2 vCPUs
- **Memory**: 4 GB RAM
- **Cost**: ~$0.0416/hour
- **500 episodes**: ~2-4 hours = $0.08-$0.17

### T3.Large Instance (Faster)
- **CPU**: 2 vCPUs
- **Memory**: 8 GB RAM
- **Cost**: ~$0.0832/hour
- **500 episodes**: ~1-2 hours = $0.08-$0.17

### Spot Instances (Cheapest)
- **Cost**: 60-90% discount
- **Risk**: Can be terminated
- **Best for**: Non-critical experiments

## Troubleshooting

### Common Issues
1. **Instance not starting**: Check IAM roles and security groups
2. **Dependencies failing**: Check user data script and Python versions
3. **Results not uploading**: Verify S3 bucket permissions
4. **High costs**: Use spot instances and set budget alerts

### Debug Commands
```bash
# Check instance status
aws ec2 describe-instance-status --instance-ids i-xxxxxxxxx

# Get console output
aws ec2 get-console-output --instance-id i-xxxxxxxxx

# Check S3 bucket contents
aws s3 ls s3://your-bucket-name/ --recursive
```

## Next Steps

1. **Set up monitoring**: Create CloudWatch dashboards
2. **Automate cleanup**: Create Lambda functions to terminate instances
3. **Scale experiments**: Use multiple instances for parallel experiments
4. **Optimize costs**: Use reserved instances for regular experiments

This setup gives you a professional AWS deployment that's scalable, cost-effective, and suitable for production research environments. 