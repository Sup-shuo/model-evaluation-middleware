# Security policy

## Supported version

Security fixes are applied to the latest published alpha. Older alpha snapshots
are not maintained.

## Reporting

Report suspected command injection, path traversal, secret disclosure, unsafe
process ownership or release-bundle issues privately through GitHub Security
Advisories for this repository. Do not include real credentials, private model
paths or proprietary result data in a public issue.

## Boundary

Adapter launchers and machine configuration are trusted operator inputs. Core
confines managed paths, validates JSON objects, avoids shell execution, redacts
known secrets and cleans only processes it owns. It does not sandbox arbitrary
third-party Adapter code and does not claim that saved results are cryptographic
evidence. Run untrusted plugins in an OS/container boundary appropriate for your
environment.
