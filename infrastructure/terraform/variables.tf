variable "aws_region" {
  description = "AWS region for all resources. us-east-1 (N. Virginia) is cheapest + has the full service matrix."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Prefix applied to every resource name (tags, SG, EIP, S3 bucket, etc)."
  type        = string
  default     = "uz-dropout"
}

variable "ssh_public_key" {
  description = "OpenSSH public key uploaded as the EC2 key pair. Generate with `ssh-keygen -t ed25519`."
  type        = string
}

variable "instance_type" {
  description = <<-EOT
    EC2 instance size.
      t3.large   — 2 vCPU /  8 GB — API + inference only, no training
      t3.xlarge  — 4 vCPU / 16 GB — recommended (comfortably runs the full bake-off)
      t3.2xlarge — 8 vCPU / 32 GB — faster training + SHAP for big cohorts
  EOT
  type    = string
  default = "t3.xlarge"
}

variable "root_volume_size_gb" {
  description = "Root EBS volume (GB). Models + MLflow DB + DVC cache fit comfortably in 30 GB."
  type        = number
  default     = 30
}

variable "allowed_cidr_blocks" {
  description = <<-EOT
    CIDR blocks allowed to reach the public app ports. `0.0.0.0/0` makes the
    dashboard public (fine for a demo). Tighten to your office IP/32 for real
    production.
  EOT
  type    = list(string)
  default = ["0.0.0.0/0"]
}

# The project can run behind either the bundled MLflow container (default)
# or against an external MLflow server. Kept as a variable so a future
# sandbox or staging environment can point at a shared MLflow without
# rebuilding the module.
variable "mlflow_artifact_bucket_suffix" {
  description = "Suffix appended to the project name for the MLflow artefact bucket. Change if the default name is taken globally."
  type        = string
  default     = "mlflow-artifacts"
}
