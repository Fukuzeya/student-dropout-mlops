locals {
  public_ip = aws_eip.uz_dropout.public_ip
}

output "ec2_public_ip" {
  description = "Elastic IP attached to the EC2 host. Stable across reboots."
  value       = local.public_ip
}

output "ssh_command" {
  description = "Copy-paste SSH command using the generated key pair."
  value       = "ssh -i ~/.ssh/${local.key_pair_name} ec2-user@${local.public_ip}"
}

output "api_url" {
  description = "FastAPI base URL."
  value       = "http://${local.public_ip}:8000"
}

output "api_docs_url" {
  description = "Interactive OpenAPI docs."
  value       = "http://${local.public_ip}:8000/docs"
}

output "frontend_url" {
  description = "Angular dashboard."
  value       = "http://${local.public_ip}:4200"
}

output "mlflow_url" {
  description = "MLflow tracking + Model Registry UI."
  value       = "http://${local.public_ip}:5000"
}

output "grafana_url" {
  description = "Grafana dashboards (default login admin/admin)."
  value       = "http://${local.public_ip}:3000"
}

output "prometheus_url" {
  description = "Prometheus metrics UI."
  value       = "http://${local.public_ip}:9090"
}

output "minio_console_url" {
  description = "MinIO admin console (S3-compatible artefact store for local dev)."
  value       = "http://${local.public_ip}:9001"
}

output "s3_bucket_name" {
  description = "S3 bucket backing MLflow artefacts in production."
  value       = aws_s3_bucket.mlflow_artifacts.bucket
}

output "deploy_command" {
  description = "Runs from inside the EC2 host to redeploy the current stack."
  value       = "cd /opt/${var.project_name} && ./scripts/deploy_aws.sh"
}
