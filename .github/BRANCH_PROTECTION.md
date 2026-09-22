# Branch protection policy

This policy applies to the shared Echelon Quant repository and must be configured in
GitHub Repository Settings → Rules → Rulesets (or Branch protection rules).

## Protected branches

- `main`: release-ready code only.
- `Develop`: shared integration branch for completed work.

Feature branches should use one of these forms:

- `feature/<area>-<short-description>`
- `fix/<area>-<short-description>`
- `chore/<short-description>`
- `docs/<short-description>`
- `ml/<short-description>`

## Required settings for both protected branches

- Require a pull request before merging.
- Require at least 1 approving review.
- Require approval from someone other than the last person who pushed.
- Dismiss stale approvals when new commits are pushed.
- Require all review conversations to be resolved.
- Require branches to be up to date before merging.
- Require these checks to pass:
  - `Lint`
  - `Type check`
  - `Tests`
  - `ML toolkit tests`
  - `Docker build`
  - `CodeQL analysis`
  - `Dependency review` for pull requests
- Block force pushes and branch deletion.
- Do not allow administrators to bypass the rules except for documented emergency recovery.
- Use squash merging as the default merge method and delete merged feature branches.

## Collaboration rules

All three contributors may create branches and open pull requests. A contributor must
not approve their own pull request. Changes affecting risk controls, order execution,
database migrations, shared event contracts, or model-serving behavior should receive
two approvals whenever the team can provide them, even though the enforced minimum is
one.

## Important GitHub setup note

This file documents the policy but GitHub does not enforce branch protection from a
repository file alone. A repository administrator must apply these settings in GitHub
and verify the required check names after the first workflow run.
