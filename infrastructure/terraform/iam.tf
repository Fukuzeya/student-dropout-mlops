# EC2 gets an instance profile so MLflow (and any other container) can
# read/write the artefacts bucket without us baking AWS keys into the
# image. Scoped to the single bucket — no broad S3 access.

data "aws_iam_policy_document" "ec2_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "s3_mlflow_access" {
  statement {
    sid = "MlflowArtifactBucketObjects"
    actions = [
      "s3:PutObject",
      "s3:GetObject",
      "s3:DeleteObject",
      "s3:ListBucket",
    ]
    resources = [
      aws_s3_bucket.mlflow_artifacts.arn,
      "${aws_s3_bucket.mlflow_artifacts.arn}/*",
    ]
  }
}

resource "aws_iam_role" "uz_dropout" {
  name               = local.role_name
  assume_role_policy = data.aws_iam_policy_document.ec2_assume_role.json
}

resource "aws_iam_role_policy" "s3_mlflow_access" {
  name   = local.role_policy_name
  role   = aws_iam_role.uz_dropout.id
  policy = data.aws_iam_policy_document.s3_mlflow_access.json
}

resource "aws_iam_instance_profile" "uz_dropout" {
  name = local.instance_profile_name
  role = aws_iam_role.uz_dropout.name
}
