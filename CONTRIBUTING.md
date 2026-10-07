# Contributing

## Branches

- `main` always holds reviewed work. Do not commit to it directly.
- Create one branch per Jira ticket: `feature/<ticket>-<short-name>`, for example `feature/103-oulad-loader`.

## Commits

- Start the message with the ticket number: `103: add loader for the seven OULAD tables`.
- Keep commits small and focused.

## Pull requests

- Open a pull request into `main` and fill in the template.
- Tag at least one teammate for review.
- Run `pytest` before opening the pull request.

## Data and secrets

- The repository is public. Never commit raw or processed data, credentials, `.env` files or personal details.
- Put data in `data/raw/`. Git ignores it.

## Code

- Reusable logic goes in `src/sar/`. Notebooks import from it and do not hold the only copy of any logic.
- Notebook names: `<ticket>_<initials>_<topic>.ipynb`.
- The Day 30 cutoff is read from `sar.config.CUTOFF_DAY`. Do not hard code 30.
- No feature may use information from after the cutoff. See `docs/leakage_checklist.md`.

## Jira

- Move a ticket to Done only when the work is merged and reviewed.
