# Single EC2 host running every platform service under Docker Compose.
# An Elastic IP is attached so the public address is stable across
# start/stop cycles — matters because the GitHub Actions deploy job SSHes
# to it by hostname.

resource "aws_eip" "uz_dropout" {
  domain = "vpc"

  tags = {
    Name = "${var.project_name}-eip"
  }
}

resource "aws_instance" "uz_dropout" {
  ami                         = data.aws_ami.amazon_linux_2023.id
  instance_type               = var.instance_type
  key_name                    = aws_key_pair.uz_dropout.key_name
  vpc_security_group_ids      = [aws_security_group.uz_dropout.id]
  iam_instance_profile        = aws_iam_instance_profile.uz_dropout.name
  associate_public_ip_address = true
  subnet_id                   = data.aws_subnets.default.ids[0]

  root_block_device {
    volume_size           = var.root_volume_size_gb
    volume_type           = "gp3"
    delete_on_termination = true
    encrypted             = true
  }

  # Idempotent bootstrap: installs Docker + Compose + git, creates the
  # deploy directory, and leaves the box ready for the GitHub Actions
  # deploy job to drop a docker-compose.prod.yml into /opt/uz-dropout.
  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    project_name = var.project_name
  })

  # Re-running user_data on every change would rebuild the box. Only
  # replace the instance when the user_data hash genuinely changes.
  user_data_replace_on_change = false

  tags = {
    Name = local.instance_name
  }

  lifecycle {
    # The AMI ID moves as Amazon publishes new AL2023 releases. Ignore
    # that drift so `terraform apply` on a new laptop doesn't try to
    # rebuild the running instance.
    ignore_changes = [ami]
  }
}

resource "aws_eip_association" "uz_dropout" {
  instance_id   = aws_instance.uz_dropout.id
  allocation_id = aws_eip.uz_dropout.id
}
