#!/bin/bash

# AWS Deployment Script for Adaptive Robot Navigation System
# This script automates the entire AWS deployment process

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
BUCKET_NAME="adaptive-robot-experiments-${TIMESTAMP}"
REGION="us-east-1"
INSTANCE_TYPE="t3.medium"
EXPERIMENT_TYPE=${1:-basic}
EPISODES=${2:-500}

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        error "AWS CLI is not installed. Please install it first."
    fi
    
    # Check if AWS is configured
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS CLI is not configured. Run 'aws configure' first."
    fi
    
    # Check if we're in the project directory
    if [[ ! -f "$PROJECT_DIR/requirements.txt" ]]; then
        error "Not in project directory. Please run from the project root."
    fi
    
    success "Prerequisites check passed"
}

# Create S3 bucket
create_s3_bucket() {
    log "Creating S3 bucket: $BUCKET_NAME"
    
    # Create bucket
    if aws s3 mb "s3://$BUCKET_NAME" --region "$REGION" 2>/dev/null; then
        success "S3 bucket created: $BUCKET_NAME"
    else
        error "Failed to create S3 bucket"
    fi
    
    # Set bucket policy for public read access to results (SKIPPED: not needed for private workflow)
    # cat > /tmp/bucket-policy.json << EOF
    # {
    #     "Version": "2012-10-17",
    #     "Statement": [
    #         {
    #             "Sid": "PublicReadGetObject",
    #             "Effect": "Allow",
    #             "Principal": "*",
    #             "Action": "s3:GetObject",
    #             "Resource": "arn:aws:s3:::$BUCKET_NAME/results/*"
    #         }
    #     ]
    # }
    # EOF
    # aws s3api put-bucket-policy --bucket "$BUCKET_NAME" --policy file:///tmp/bucket-policy.json
    # success "Bucket policy set"
}

# Create IAM role
create_iam_role() {
    log "Creating IAM role for EC2..."
    
    # Check if role already exists
    if aws iam get-role --role-name ExperimentRunnerRole &> /dev/null; then
        warning "IAM role ExperimentRunnerRole already exists"
        return 0
    fi
    
    # Create trust policy
    cat > /tmp/trust-policy.json << EOF
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
    aws iam create-role --role-name ExperimentRunnerRole --assume-role-policy-document file:///tmp/trust-policy.json > /dev/null
    log "IAM role created, attaching policies..."
    
    # Attach policies
    aws iam attach-role-policy --role-name ExperimentRunnerRole --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
    aws iam attach-role-policy --role-name ExperimentRunnerRole --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess
    
    # Create instance profile
    aws iam create-instance-profile --instance-profile-name ExperimentRunnerProfile
    aws iam add-role-to-instance-profile --instance-profile-name ExperimentRunnerProfile --role-name ExperimentRunnerRole
    
    # Wait for role to be available
    log "Waiting for IAM role to be available..."
    sleep 10
    
    success "IAM role created"
}

# Prepare project for upload
prepare_project() {
    log "Preparing project for upload..."
    
    cd "$PROJECT_DIR"
    
    # Create project archive (excluding unnecessary files)
    if [[ -f "project.zip" ]]; then
        rm project.zip
    fi
    
    zip -r project.zip . \
        -x "*.git*" \
        -x "*.pyc" \
        -x "__pycache__/*" \
        -x "*.DS_Store" \
        -x "data/experiments/*" \
        -x ".pytest_cache/*" \
        -x "adaptive_robot_navigation.egg-info/*" \
        -x "results/*" \
        -x "*.log" \
        -x "*.tmp"
    
    success "Project archive created: project.zip"
}

# Upload project to S3
upload_project() {
    log "Uploading project to S3..."
    
    aws s3 cp project.zip "s3://$BUCKET_NAME/project.zip"
    success "Project uploaded to S3"
}

# Create user data script
create_user_data() {
    log "Creating user data script..."
    
    cat > /tmp/user-data.sh << 'EOF'
#!/bin/bash

# Log everything to a file
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "Starting user data script..."

# Update system
yum update -y
yum install -y python3 python3-pip git unzip

# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install

# Get project from S3
aws s3 cp s3://BUCKET_NAME/project.zip /home/ec2-user/
cd /home/ec2-user
unzip -o project.zip

# Install system dependencies for OpenCV
yum install -y mesa-libGL mesa-libGL-devel mesa-libGLU mesa-libGLU-devel libXext libXrender libXtst libXi

# Install Python dependencies
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install numpy scipy matplotlib seaborn pyyaml gym stable-baselines3 opencv-python tqdm pandas scikit-learn pytest

# Install project
cd "Project 1 - Adaptive Robot Navigation System"
pip3 install -e .

# Create results directory
mkdir -p data/experiments

# Run experiments with detailed error capture
echo "Starting experiment: EXPERIMENT_TYPE with EPISODES episodes"
echo "Current directory: $(pwd)"
echo "Python version: $(python3 --version)"
echo "Pip list:"
pip3 list

# Capture experiment output and errors
echo "Running experiment..."
if python3 cloud_setup/headless_experiment.py --experiment EXPERIMENT_TYPE --episodes EPISODES 2>&1 | tee /tmp/experiment_output.log; then
    echo "Experiment completed successfully"
else
    echo "Experiment failed with exit code $?"
    echo "Experiment error output:"
    cat /tmp/experiment_output.log
fi

# Check what files were created
echo "Files in data/experiments/:"
ls -la data/experiments/ || echo "data/experiments/ directory not found"

echo "Files in current directory:"
ls -la

echo "Files in data/:"
ls -la data/ || echo "data/ directory not found"

# Upload experiment log to S3
aws s3 cp /tmp/experiment_output.log s3://BUCKET_NAME/results/experiment_output.log

# Upload results to S3 (if they exist)
echo "Uploading results to S3..."
if [ -d "data/experiments" ]; then
    aws s3 cp data/experiments/ s3://BUCKET_NAME/results/ --recursive
    echo "Results uploaded successfully"
else
    echo "No results directory found"
fi

# Create completion marker
echo "Experiment completed at $(date)" > /tmp/experiment_complete.txt
aws s3 cp /tmp/experiment_complete.txt s3://BUCKET_NAME/results/

echo "User data script completed. Shutting down in 60 seconds..."
sleep 60
shutdown -h now
EOF
    
    # Replace placeholders
    sed -i.bak "s/BUCKET_NAME/$BUCKET_NAME/g" /tmp/user-data.sh
    sed -i.bak "s/EXPERIMENT_TYPE/$EXPERIMENT_TYPE/g" /tmp/user-data.sh
    sed -i.bak "s/EPISODES/$EPISODES/g" /tmp/user-data.sh
    rm -f /tmp/user-data.sh.bak
    
    success "User data script created"
}

# Get latest Ubuntu AMI
get_ami_id() {
    log "Getting latest Ubuntu AMI..." >&2
    # Use a known Ubuntu 20.04 AMI for us-east-1
    AMI_ID="ami-0c02fb55956c7d316"
    success "Using AMI: $AMI_ID" >&2
    echo "$AMI_ID"
}

# Create launch template
create_launch_template() {
    log "Creating launch template..."
    
    AMI_ID=$(get_ami_id)
    
    # Create launch template data file
    cat > /tmp/launch-template-data.json << EOF
{
  "ImageId": "$AMI_ID",
  "InstanceType": "$INSTANCE_TYPE",
  "IamInstanceProfile": {
    "Name": "ExperimentRunnerProfile"
  },
  "UserData": "$(base64 < /tmp/user-data.sh | tr -d '\n')"
}
EOF
    
    # Check if template already exists
    if aws ec2 describe-launch-templates --launch-template-names ExperimentRunnerTemplate &> /dev/null; then
        log "Launch template already exists, creating new version..."
        aws ec2 create-launch-template-version \
            --launch-template-name ExperimentRunnerTemplate \
            --version-description "v$(date +%s)" \
            --launch-template-data file:///tmp/launch-template-data.json
    else
        aws ec2 create-launch-template \
            --launch-template-name ExperimentRunnerTemplate \
            --version-description v1 \
            --launch-template-data file:///tmp/launch-template-data.json
    fi
    
    success "Launch template created"
}

# Launch EC2 instance
launch_instance() {
    log "Launching EC2 instance..." >&2
    
    # Get default VPC and subnet
    VPC_ID=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" --query "Vpcs[0].VpcId" --output text)
    SUBNET_ID=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query "Subnets[0].SubnetId" --output text)
    
    # Create security group if it doesn't exist
    SG_NAME="ExperimentRunnerSG"
    SG_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" --query "SecurityGroups[0].GroupId" --output text)
    
    if [[ "$SG_ID" == "None" ]]; then
        log "Creating security group..." >&2
        SG_ID=$(aws ec2 create-security-group \
            --group-name "$SG_NAME" \
            --description "Security group for experiment runner" \
            --vpc-id "$VPC_ID" \
            --query "GroupId" \
            --output text)
        
        # Allow outbound traffic
        aws ec2 authorize-security-group-egress \
            --group-id "$SG_ID" \
            --protocol -1 \
            --port -1 \
            --cidr 0.0.0.0/0
    fi
    
    # Launch instance
    INSTANCE_ID=$(aws ec2 run-instances \
        --launch-template LaunchTemplateName=ExperimentRunnerTemplate,Version=\$Latest \
        --instance-type "$INSTANCE_TYPE" \
        --security-group-ids "$SG_ID" \
        --subnet-id "$SUBNET_ID" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=ExperimentRunner-$TIMESTAMP}]" \
        --query "Instances[0].InstanceId" \
        --output text)
    
    if [[ -z "$INSTANCE_ID" ]]; then
        error "Failed to launch instance" >&2
    fi
    
    success "Instance launched: $INSTANCE_ID" >&2
    echo "$INSTANCE_ID"
}

# Monitor experiment progress
monitor_progress() {
    local INSTANCE_ID=$1
    
    log "Monitoring experiment progress..."
    log "Instance ID: $INSTANCE_ID"
    log "S3 Bucket: $BUCKET_NAME"
    log "Check progress: aws s3 ls s3://$BUCKET_NAME/results/"
    
    echo ""
    echo "=== Monitoring Commands ==="
    echo "Check instance status:"
    echo "  aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].{State:State.Name,PublicIP:PublicIpAddress}'"
    echo ""
    echo "Check experiment progress:"
    echo "  aws s3 ls s3://$BUCKET_NAME/results/"
    echo ""
    echo "Download results when complete:"
    echo "  aws s3 sync s3://$BUCKET_NAME/results/ ./results/"
    echo ""
    echo "Get instance console output:"
    echo "  aws ec2 get-console-output --instance-id $INSTANCE_ID"
    echo ""
    
    # Start monitoring loop
    echo "Starting automatic monitoring (press Ctrl+C to stop)..."
    while true; do
        # Check if experiment is complete
        if aws s3 ls "s3://$BUCKET_NAME/results/experiment_complete.txt" &> /dev/null; then
            success "Experiment completed!"
            break
        fi
        
        # Check instance status
        STATE=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --query "Reservations[0].Instances[0].State.Name" --output text)
        if [[ "$STATE" == "terminated" ]]; then
            warning "Instance terminated"
            break
        fi
        
        # Check for new results
        RESULT_COUNT=$(aws s3 ls "s3://$BUCKET_NAME/results/" 2>/dev/null | wc -l)
        echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} Instance: $STATE, Results: $RESULT_COUNT files"
        
        sleep 30
    done
    
    # Download and display experiment logs
    download_experiment_logs
}

# Download and display experiment logs
download_experiment_logs() {
    log "Downloading experiment logs..."
    
    # Download experiment output log
    if aws s3 cp "s3://$BUCKET_NAME/results/experiment_output.log" /tmp/experiment_output.log &> /dev/null; then
        echo ""
        echo "=== EXPERIMENT OUTPUT LOG ==="
        cat /tmp/experiment_output.log
        echo ""
    else
        warning "No experiment output log found"
    fi
    
    # Download all results
    aws s3 sync "s3://$BUCKET_NAME/results/" ./results/
    
    echo ""
    echo "=== DOWNLOADED RESULTS ==="
    ls -la ./results/
    echo ""
    
    # Display completion info
    if [[ -f "./results/experiment_complete.txt" ]]; then
        echo "=== EXPERIMENT COMPLETION ==="
        cat ./results/experiment_complete.txt
        echo ""
    fi
}

# Main deployment function
deploy_experiment() {
    log "Starting AWS deployment for experiment: $EXPERIMENT_TYPE with $EPISODES episodes"
    
    check_prerequisites
    create_s3_bucket
    create_iam_role
    prepare_project
    upload_project
    create_user_data
    create_launch_template
    
    INSTANCE_ID=$(launch_instance)
    
    success "Deployment completed successfully!"
    success "Instance ID: $INSTANCE_ID"
    success "S3 Bucket: $BUCKET_NAME"
    
    # Save deployment info
    cat > deployment_info.txt << EOF
Deployment Information:
======================
Timestamp: $(date)
Instance ID: $INSTANCE_ID
S3 Bucket: $BUCKET_NAME
Experiment Type: $EXPERIMENT_TYPE
Episodes: $EPISODES
Region: $REGION
Instance Type: $INSTANCE_TYPE

Monitoring Commands:
===================
Check status: aws ec2 describe-instances --instance-ids $INSTANCE_ID
Check progress: aws s3 ls s3://$BUCKET_NAME/results/
Download results: aws s3 sync s3://$BUCKET_NAME/results/ ./results/
EOF
    
    success "Deployment info saved to deployment_info.txt"
    
    # Start monitoring
    monitor_progress "$INSTANCE_ID"
    
    # Check for common issues
    check_experiment_issues
}

# Cleanup function
cleanup() {
    log "Cleaning up temporary files..."
    rm -f /tmp/bucket-policy.json
    rm -f /tmp/trust-policy.json
    rm -f /tmp/user-data.sh
    rm -f /tmp/launch-template-data.json
    rm -f /tmp/experiment_output.log
    success "Cleanup completed"
}

# Check for common experiment issues
check_experiment_issues() {
    log "Checking for common experiment issues..."
    
    # Check if experiment output log exists
    if [[ -f "./results/experiment_output.log" ]]; then
        echo ""
        echo "=== EXPERIMENT ANALYSIS ==="
        
        # Check for import errors
        if grep -i "import.*error\|module.*not found\|no module named" ./results/experiment_output.log; then
            echo "❌ IMPORT ERRORS DETECTED"
            echo "   The experiment failed due to missing Python modules."
            echo "   Check the requirements.txt and ensure all dependencies are installed."
        fi
        
        # Check for file permission errors
        if grep -i "permission.*denied\|access.*denied" ./results/experiment_output.log; then
            echo "❌ PERMISSION ERRORS DETECTED"
            echo "   The experiment failed due to file permission issues."
        fi
        
        # Check for memory errors
        if grep -i "memory.*error\|out of memory\|killed" ./results/experiment_output.log; then
            echo "❌ MEMORY ERRORS DETECTED"
            echo "   The experiment was killed due to insufficient memory."
            echo "   Consider using a larger instance type (e.g., t3.large)."
        fi
        
        # Check for successful completion
        if grep -i "experiment completed\|final success rate\|average reward" ./results/experiment_output.log; then
            echo "✅ EXPERIMENT COMPLETED SUCCESSFULLY"
        fi
        
        echo ""
    fi
}

# Trap to ensure cleanup on exit
trap cleanup EXIT

# Show usage if no arguments
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 [basic|comparison] [episodes]"
    echo "Examples:"
    echo "  $0 basic 500      # Run basic experiment with 500 episodes"
    echo "  $0 comparison 100 # Run comparison experiment with 100 episodes"
    exit 1
fi

# Validate arguments
if [[ "$EXPERIMENT_TYPE" != "basic" && "$EXPERIMENT_TYPE" != "comparison" ]]; then
    error "Experiment type must be 'basic' or 'comparison'"
fi

if ! [[ "$EPISODES" =~ ^[0-9]+$ ]]; then
    error "Episodes must be a number"
fi

# Run deployment
deploy_experiment 