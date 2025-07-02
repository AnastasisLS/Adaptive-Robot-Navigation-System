# AWS Quick Start Guide

## 🚀 Get Your Experiments Running in 5 Minutes

This guide will get your Adaptive Robot Navigation experiments running on AWS EC2 without keeping your computer on.

## Prerequisites (One-time setup)

### 1. Install AWS CLI
```bash
# macOS
brew install awscli

# Or download from AWS website
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /
```

### 2. Configure AWS CLI
```bash
aws configure
```
Enter your AWS credentials when prompted:
- **AWS Access Key ID**: Get from AWS Console → IAM → Users → Your User → Security credentials
- **AWS Secret Access Key**: Same place as above
- **Default region**: `us-east-1` (recommended)
- **Output format**: `json`

## 🎯 Quick Start (3 Steps)

### Step 1: Run the Deployment Script
```bash
# Run basic experiment (500 episodes)
./cloud_setup/aws_deploy.sh basic 500

# Or run comparison experiment (100 episodes)
./cloud_setup/aws_deploy.sh comparison 100
```

### Step 2: Monitor Progress
The script will automatically monitor your experiment and show you:
- Instance status
- Progress updates every 30 seconds
- S3 bucket where results are stored

### Step 3: Download Results
```bash
# Download results when complete
aws s3 sync s3://your-bucket-name/results/ ./results/
```

## 📊 What You Get

### Automatic Setup
- ✅ S3 bucket for result storage
- ✅ EC2 instance with all dependencies
- ✅ IAM roles and security groups
- ✅ Automatic experiment execution
- ✅ Results uploaded to S3
- ✅ Instance auto-shutdown when done

### Results Include
- 📈 Training curves and metrics
- 📊 Success rates and performance data
- 🗂️ Per-episode and per-step logs
- 📁 Trajectory data for analysis
- 📋 JSON files with detailed statistics

## 💰 Cost Estimation

### T3.Medium Instance (Recommended)
- **Cost**: ~$0.0416/hour
- **500 episodes**: ~2-4 hours = **$0.08-$0.17**
- **1000 episodes**: ~4-8 hours = **$0.17-$0.33**

### Free Tier
- **AWS Free Tier**: 750 hours/month of t2.micro
- **Perfect for**: Testing and small experiments

## 🔧 Advanced Usage

### Custom Experiments
```bash
# Run with custom parameters
./cloud_setup/aws_deploy.sh basic 1000  # 1000 episodes
./cloud_setup/aws_deploy.sh comparison 200  # 200 episodes
```

### Monitor Multiple Experiments
```bash
# Check all running instances
aws ec2 describe-instances --filters "Name=tag:Name,Values=ExperimentRunner*" --query "Reservations[].Instances[].{InstanceId:InstanceId,State:State.Name,Name:Tags[?Key=='Name'].Value|[0]}"

# Check specific instance
aws ec2 describe-instances --instance-ids i-xxxxxxxxx
```

### Download Results from Any Experiment
```bash
# List all experiment buckets
aws s3 ls | grep adaptive-robot-experiments

# Download specific experiment results
aws s3 sync s3://adaptive-robot-experiments-20241201_143022/results/ ./results/
```

## 🛠️ Troubleshooting

### Common Issues

**"AWS CLI not configured"**
```bash
aws configure
# Enter your credentials
```

**"Permission denied"**
```bash
# Make script executable
chmod +x cloud_setup/aws_deploy.sh
```

**"Instance not starting"**
```bash
# Check instance status
aws ec2 describe-instances --instance-ids i-xxxxxxxxx

# Get console output
aws ec2 get-console-output --instance-id i-xxxxxxxxx
```

**"Results not uploading"**
```bash
# Check S3 bucket
aws s3 ls s3://your-bucket-name/results/
```

### Get Help
```bash
# Show script usage
./cloud_setup/aws_deploy.sh

# Check AWS CLI version
aws --version

# Test AWS connection
aws sts get-caller-identity
```

## 📈 Example Output

```
[2024-12-01 14:30:22] Checking prerequisites...
[SUCCESS] Prerequisites check passed
[2024-12-01 14:30:23] Creating S3 bucket: adaptive-robot-experiments-20241201_143022
[SUCCESS] S3 bucket created: adaptive-robot-experiments-20241201_143022
[2024-12-01 14:30:25] Creating IAM role for EC2...
[SUCCESS] IAM role created
[2024-12-01 14:30:35] Preparing project for upload...
[SUCCESS] Project archive created: project.zip
[2024-12-01 14:30:40] Uploading project to S3...
[SUCCESS] Project uploaded to S3
[2024-12-01 14:30:45] Creating user data script...
[SUCCESS] User data script created
[2024-12-01 14:30:50] Getting latest Ubuntu AMI...
[SUCCESS] Using AMI: ami-0c02fb55956c7d316
[2024-12-01 14:30:55] Creating launch template...
[SUCCESS] Launch template created
[2024-12-01 14:31:00] Launching EC2 instance...
[SUCCESS] Instance launched: i-0a1b2c3d4e5f6g7h8
[SUCCESS] Deployment completed successfully!
[SUCCESS] Instance ID: i-0a1b2c3d4e5f6g7h8
[SUCCESS] S3 Bucket: adaptive-robot-experiments-20241201_143022

=== Monitoring Commands ===
Check instance status:
  aws ec2 describe-instances --instance-ids i-0a1b2c3d4e5f6g7h8
Check experiment progress:
  aws s3 ls s3://adaptive-robot-experiments-20241201_143022/results/
Download results when complete:
  aws s3 sync s3://adaptive-robot-experiments-20241201_143022/results/ ./results/

Starting automatic monitoring (press Ctrl+C to stop)...
[14:31:30] Instance: running, Results: 0 files
[14:32:00] Instance: running, Results: 5 files
[14:32:30] Instance: running, Results: 12 files
...
[SUCCESS] Experiment completed!
```

## 🎓 Learning AWS Skills

This setup teaches you:
- **EC2**: Virtual machines in the cloud
- **S3**: Object storage for results
- **IAM**: Identity and access management
- **CloudWatch**: Monitoring and logging
- **CLI**: Command-line automation
- **Infrastructure as Code**: Automated deployment

## 🚀 Next Steps

1. **Scale up**: Run multiple experiments in parallel
2. **Optimize costs**: Use spot instances for 60-90% savings
3. **Add monitoring**: Create CloudWatch dashboards
4. **Automate cleanup**: Set up Lambda functions
5. **Production deployment**: Use AWS Batch for large-scale experiments

## 📞 Support

- **AWS Documentation**: https://docs.aws.amazon.com/
- **AWS Free Tier**: https://aws.amazon.com/free/
- **Project Issues**: Check the main README.md

---

**Ready to run your first cloud experiment?** 🚀

```bash
./cloud_setup/aws_deploy.sh basic 500
``` 