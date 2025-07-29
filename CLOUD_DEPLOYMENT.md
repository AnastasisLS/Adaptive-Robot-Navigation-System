# AWS Cloud Deployment Guide for Adaptive Robot Navigation System

## Overview
This guide provides comprehensive instructions for deploying experiments to AWS EC2, enabling long-running experiments without requiring local computational resources. The deployment utilizes AWS EC2 with auto-scaling and S3 for result storage.

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
# Get latest Ubuntu AMI
AMI_ID=$(aws ec2 describe-images \
  --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*" \
  --query "sort_by(Images, &CreationDate)[-1].ImageId" \
  --output text)

# Create launch template data
cat > launch-template-data.json << EOF
{
  "ImageId": "$AMI_ID",
  "InstanceType": "t3.medium",
  "IamInstanceProfile": {
    "Name": "ExperimentRunnerProfile"
  },
  "SecurityGroupIds": ["sg-xxxxxxxxx"],
  "UserData": "$(base64 -w 0 user-data.sh)",
  "TagSpecifications": [
    {
      "ResourceType": "instance",
      "Tags": [
        {
          "Key": "Name",
          "Value": "ExperimentRunner"
        },
        {
          "Key": "Project",
          "Value": "AdaptiveRobotNavigation"
        }
      ]
    }
  ]
}
EOF

# Create launch template
aws ec2 create-launch-template \
  --launch-template-name ExperimentRunnerTemplate \
  --launch-template-data file://launch-template-data.json
```

### Launch Instance
```bash
# Launch instance using template
aws ec2 run-instances \
  --launch-template LaunchTemplateName=ExperimentRunnerTemplate,Version=\$Latest \
  --count 1 \
  --query 'Instances[0].InstanceId' \
  --output text
```

## Step 5: Monitor and Download Results

### Monitor Instance Status
```bash
# Check instance status
aws ec2 describe-instances \
  --instance-ids i-xxxxxxxxx \
  --query 'Reservations[0].Instances[0].State.Name' \
  --output text
```

### Download Results
```bash
# Download results when experiment completes
aws s3 sync s3://adaptive-robot-experiments-$(date +%s)/results/ ./results/
```

## Advanced Configuration

### Security Groups
```bash
# Create security group
aws ec2 create-security-group \
  --group-name ExperimentRunnerSG \
  --description "Security group for experiment runners"

# Allow SSH access (optional)
aws ec2 authorize-security-group-ingress \
  --group-name ExperimentRunnerSG \
  --protocol tcp \
  --port 22 \
  --cidr 0.0.0.0/0
```

### Auto Scaling (Optional)
```bash
# Create auto scaling group for multiple experiments
aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name ExperimentRunnerASG \
  --launch-template LaunchTemplateName=ExperimentRunnerTemplate,Version=\$Latest \
  --min-size 0 \
  --max-size 10 \
  --desired-capacity 1 \
  --vpc-zone-identifier subnet-xxxxxxxxx
```

## Cost Optimization

### Spot Instances
```bash
# Use spot instances for cost savings
aws ec2 run-instances \
  --launch-template LaunchTemplateName=ExperimentRunnerTemplate,Version=\$Latest \
  --instance-market-options MarketType=spot,SpotOptions={MaxPrice=0.05} \
  --count 1
```

### Instance Types
- **t3.micro**: Free tier, suitable for small experiments
- **t3.small**: $0.0208/hour, good for medium experiments
- **t3.medium**: $0.0416/hour, recommended for most experiments
- **c5.large**: $0.085/hour, for compute-intensive experiments

## Troubleshooting

### Common Issues

1. **Instance fails to start**
   - Verify IAM role permissions
   - Check security group configuration
   - Ensure sufficient quota

2. **Experiment fails to run**
   - Check S3 bucket permissions
   - Verify user data script
   - Review CloudWatch logs

3. **Results not uploaded**
   - Check S3 bucket policy
   - Verify IAM role has S3 permissions
   - Review experiment logs

### Debug Commands
```bash
# Get instance console output
aws ec2 get-console-output --instance-id i-xxxxxxxxx

# Check S3 bucket contents
aws s3 ls s3://your-bucket-name/results/

# Monitor instance in real-time
aws ec2 describe-instances --instance-ids i-xxxxxxxxx --query 'Reservations[0].Instances[0].{State:State.Name,PublicIP:PublicIpAddress}'
```

## Best Practices

1. **Security**
   - Use IAM roles with minimal permissions
   - Implement proper security groups
   - Regularly rotate access keys

2. **Cost Management**
   - Use spot instances when possible
   - Set up billing alerts
   - Terminate instances promptly

3. **Reliability**
   - Implement proper error handling
   - Use CloudWatch for monitoring
   - Backup important results

4. **Scalability**
   - Use auto scaling for multiple experiments
   - Implement proper resource tagging
   - Monitor resource usage

## Cleanup

### Terminate Resources
```bash
# Terminate instance
aws ec2 terminate-instances --instance-ids i-xxxxxxxxx

# Delete launch template
aws ec2 delete-launch-template --launch-template-name ExperimentRunnerTemplate

# Delete IAM role (after detaching from instance profile)
aws iam remove-role-from-instance-profile --instance-profile-name ExperimentRunnerProfile --role-name ExperimentRunnerRole
aws iam delete-instance-profile --instance-profile-name ExperimentRunnerProfile
aws iam delete-role --role-name ExperimentRunnerRole

# Delete S3 bucket (after removing all objects)
aws s3 rb s3://your-bucket-name --force
``` 