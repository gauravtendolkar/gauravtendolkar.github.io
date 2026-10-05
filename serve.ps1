# Local preview at http://localhost:4000 (needs Docker Desktop running).
# Edits under docs/ are picked up automatically; refresh the browser.
$ErrorActionPreference = 'Stop'
$site = Join-Path $PSScriptRoot 'docs'
docker rm -f gt-jekyll 2>$null | Out-Null
docker run -d --name gt-jekyll -p 4000:4000 `
  -v "${site}:/srv/site" -v gt-jekyll-gems:/usr/local/bundle -w /srv/site ruby:3.3 `
  bash -lc "bundle install --quiet && bundle exec jekyll serve --host 0.0.0.0 --port 4000 --force_polling" | Out-Null
Write-Host "Starting... open http://localhost:4000  (logs: docker logs -f gt-jekyll, stop: docker rm -f gt-jekyll)"
