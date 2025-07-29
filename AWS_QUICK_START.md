# AWS Quick Start Guide

## Rapid Experiment Deployment

This guide provides instructions for deploying Adaptive Robot Navigation experiments on AWS EC2 without requiring local computational resources.

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

## Quick Start (3 Steps)

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

## System Features

### Automatic Setup
- S3 bucket for result storage
- EC2 instance with all dependencies
- IAM roles and security groups
- Automatic experiment execution
- Results uploaded to S3
- Instance auto-shutdown when done

### Results Include
- Training curves and metrics
- Success rates and performance data
- Per-episode and per-step logs
- Trajectory data for analysis
- JSON files with detailed statistics

## Cost Estimation

### T3.Medium Instance (Recommended)
- **Cost**: ~$0.0416/hour
- **500 episodes**: ~2-4 hours = **$0.08-$0.17**
- **1000 episodes**: ~4-8 hours = **$0.17-$0.33**

### Free Tier
- **AWS Free Tier**: 750 hours/month of t2.micro
- **Perfect for**: Testing and small experiments

## Advanced Usage

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
aws s3 sync s3://adaptive-robot-experiments-YYYYMMDDHHMMSS/results/ ./results/
```

## Troubleshooting

### Common Issues

1. **Instance fails to start**
   - Check AWS credentials and permissions
   - Verify region availability
   - Ensure sufficient quota for EC2 instances

2. **Experiment fails to run**
   - Check S3 bucket permissions
   - Verify IAM role has necessary permissions
   - Review experiment logs in S3

3. **Results not downloaded**
   - Verify S3 bucket name
   - Check local directory permissions
   - Ensure experiment completed successfully

### Debug Commands

```bash
# Check instance status
aws ec2 describe-instances --instance-ids i-xxxxxxxxx

# View experiment logs
aws s3 cp s3://your-bucket/results/experiment_output.log -

# Monitor instance in real-time
aws ec2 get-console-output --instance-id i-xxxxxxxxx
```

## Security Considerations

- All instances use IAM roles with minimal required permissions
- Security groups restrict access to necessary ports only
- S3 buckets are private by default
- Instances auto-terminate after experiment completion
- No persistent storage of sensitive data

## Best Practices

1. **Cost Management**
   - Use spot instances for cost optimization
   - Set up billing alerts
   - Monitor instance usage

2. **Experiment Design**
   - Start with small experiments to validate setup
   - Use appropriate instance types for workload
   - Implement proper error handling

3. **Data Management**
   - Regularly backup important results
   - Use descriptive bucket names
   - Clean up old experiments periodically 