#!/usr/bin/env bash
# Usage: ./set_github_user.sh YOUR_GITHUB_USERNAME
# Replaces the USERNAME placeholder in the repository metadata and commits the change.
set -euo pipefail
if [ $# -ne 1 ]; then echo "usage: $0 YOUR_GITHUB_USERNAME"; exit 1; fi
cd "$(dirname "$0")"
sed -i.bak "s#github.com/USERNAME/#github.com/$1/#g" CITATION.cff README.md && rm -f CITATION.cff.bak README.md.bak
git add CITATION.cff README.md
git commit -m "Set GitHub repository URL" || true
echo "Done. Repository URL: https://github.com/$1/vae-palns-shift"
