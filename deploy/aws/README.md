# AWS GPU HTTP smoke test

These files define the account-scoped IAM and bootstrap inputs for the initial
`eu-north-1` test. The container image is pinned by ECR digest in
`user-data.sh`; update that digest deliberately for a later image.

The EC2 instance profile combines:

- the inline permissions in `worker-s3-policy.json`;
- `AmazonEC2ContainerRegistryPullOnly` for the private image;
- `AmazonSSMManagedInstanceCore` for administration without SSH.

The Launch Template must require IMDSv2 and set its response hop limit to `2`
so the bridged Docker container can obtain temporary instance-role credentials.
No access key belongs in user data, the image, or an HTTP request.
Leave `WAKE_S3_ENDPOINT_URL` unset on AWS so boto3 uses the native AWS S3
endpoint and instance-role credential chain.

The HTTP security-group rule is temporary and restricted to the test client's
single public IPv4 address. The GPU instance is terminated after the smoke test;
ECR and S3 remain until explicitly cleaned up.
