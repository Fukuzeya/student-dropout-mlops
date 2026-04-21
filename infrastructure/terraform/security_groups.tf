# Single SG for every platform port. Demo-grade: tighten `allowed_cidr_blocks`
# to an office IP before taking this anywhere near real students' data.
#
# Ports (all TCP):
#   22   — SSH
#   3000 — Grafana dashboards
#   4200 — Angular frontend
#   5000 — MLflow tracking server
#   8000 — FastAPI prediction API
#   9000 — MinIO S3 API
#   9001 — MinIO console
#   9090 — Prometheus

resource "aws_security_group" "uz_dropout" {
  name        = local.security_group_name
  description = "Student Dropout Early Warning System - allow platform service ports"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  ingress {
    description = "FastAPI prediction API"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  ingress {
    description = "Angular frontend"
    from_port   = 4200
    to_port     = 4200
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  ingress {
    description = "MLflow tracking server"
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  ingress {
    description = "Grafana dashboards"
    from_port   = 3000
    to_port     = 3000
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  ingress {
    description = "Prometheus metrics"
    from_port   = 9090
    to_port     = 9090
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  ingress {
    description = "MinIO S3 API"
    from_port   = 9000
    to_port     = 9000
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  ingress {
    description = "MinIO web console"
    from_port   = 9001
    to_port     = 9001
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  egress {
    description = "All outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = local.security_group_name
  }
}
