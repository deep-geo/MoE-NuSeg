## Repository Branches

### `main` Branch
- Configured for 4-channel RGB + SWT input.
- Includes SWT preprocessing and models adapted for the additional channel.

### `rgb-only` Branch
- Configured for 3-channel RGB input (baseline setup).
- Contains models and scripts for experiments without SWT.

### How to Switch Branches
To switch between branches, use:
```bash
# For RGB-only experiments
git checkout rgb-only

# For SWT-enhanced experiments
git checkout main