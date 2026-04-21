resource "aws_key_pair" "uz_dropout" {
  key_name   = local.key_pair_name
  public_key = var.ssh_public_key
}
