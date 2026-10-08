$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path upstream | Out-Null
if (-not (Test-Path upstream/lc0/.git)) {
  git clone --recursive https://github.com/LeelaChessZero/lc0.git upstream/lc0
}
if (-not (Test-Path upstream/maia3/.git)) {
  git clone https://github.com/CSSLab/maia3.git upstream/maia3
}
Write-Host "Upstream sources ready. This script does not yet integrate Maia3 inference."
