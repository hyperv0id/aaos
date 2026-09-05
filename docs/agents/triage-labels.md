# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the actual label strings used in this repo's issue tracker.

| Label in mattpocock/skills | Label in our tracker | Meaning                                  |
| -------------------------- | -------------------- | ---------------------------------------- |
| `needs-triage`             | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`               | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent`          | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human`          | `ready-for-human`    | Requires human implementation            |
| `wontfix`                  | `wontfix`            | Will not be actioned                     |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the corresponding label string from this table.

Edit the right-hand column to match whatever vocabulary you actually use.

## Repo-label vocabulary

Beyond the five canonical triage labels, the repo's automation applies these labels. They are
not triage roles — don't map them onto the table above; just be aware when triaging:

| Label | Applied by | Meaning |
| ----- | ---------- | ------- |
| `bug` | issue template `bug_report.yml` | Confirmed defect report (auto-applied at creation, alongside `needs-triage`) |
| `enhancement` | issue template `feature_request.yml` | Feature request |
| `dependencies` | `.github/dependabot.yml` | Dependabot dependency-bump PR |
| `ci` | `.github/dependabot.yml` | Dependabot GitHub-Actions-bump PR |

Note: `feature_request.yml` does not auto-apply `needs-triage` — add it manually when triaging
feature requests so they enter the same pipeline as bugs.
