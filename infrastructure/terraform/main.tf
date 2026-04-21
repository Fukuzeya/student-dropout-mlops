locals {
  name_prefix          = var.project_name
  mlflow_bucket_name   = "${var.project_name}-${var.mlflow_artifact_bucket_suffix}"
  instance_name        = "${var.project_name}-ec2"
  security_group_name  = "${var.project_name}-sg"
  key_pair_name        = "${var.project_name}-key"
  instance_profile_name = "${var.project_name}-instance-profile"
  role_name            = "${var.project_name}-ec2-role"
  role_policy_name     = "${var.project_name}-s3-policy"
}

# Amazon Linux 2023 — current-gen, free, patched by AWS.
# Using a data source (not a hard-coded AMI ID) keeps the module portable
# across regions: the owner 137112412989 is Amazon's canonical AL2023 publisher.
data "aws_ami" "amazon_linux_2023" {
  most_recent = true
  owners      = ["137112412989"]

  filter {
    name   = "name"
    values = ["al2023-ami-ecs-hvm-*-x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# The default VPC is enough for a single-node demo — saves us a whole
# networking module. Production workloads should obviously move to a
# purpose-built VPC with private subnets + NAT.
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}
