# Policy Compliance Notes

## YouTube

Nexisgen operators must review and comply with YouTube API and developer policies
before running large-scale ingestion jobs.

Reference:

- https://developers.google.com/youtube/terms/developer-policies-guide

Key implications:

- do not assume unrestricted downloading rights
- avoid collecting personal/sensitive user data
- keep clear provenance metadata for every clip
- maintain an operator review gate for disputed content

## Hippius

Use dedicated access keys and rotate credentials regularly.

References:

- https://docs.hippius.com/cli/usage
- https://docs.hippius.com/storage/s3/integration

