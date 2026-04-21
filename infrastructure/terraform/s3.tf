# MLflow artefact store. Private bucket, versioned so a bad promotion can
# always be rolled back by pointing MLflow at the previous version.

resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = local.mlflow_bucket_name
}

resource "aws_s3_bucket_versioning" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "mlflow_artifacts" {
  bucket                  = aws_s3_bucket.mlflow_artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
