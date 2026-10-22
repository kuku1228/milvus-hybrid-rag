<#
    push_to_github.ps1  -  one-shot helper to init / commit / push this project

    Usage:
        powershell -ExecutionPolicy Bypass -File .\push_to_github.ps1 -RepoUrl https://github.com/<user>/<repo>.git
        powershell -ExecutionPolicy Bypass -File .\push_to_github.ps1 -RepoUrl https://github.com/<user>/<repo>.git -ExcludePdf

    Notes:
        * Safe to re-run: init / remote / commit steps are idempotent.
        * This script excludes itself via .git/info/exclude, so it will never be committed.
        * -ExcludePdf keeps data/*.pdf out of the repo (local-only, does not touch .gitignore).
        * All output is ASCII on purpose: Windows PowerShell 5.1 reads BOM-less UTF-8 files as GBK.
#>
[CmdletBinding()]
param(
    [string]$RepoUrl = "",
    [string]$ProjectPath = "D:\Python\PycharmProjects\RAG",
    [switch]$ExcludePdf
)

if (Get-Variable PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

function Info([string]$m) { Write-Host "[*] $m" -ForegroundColor Cyan }
function Ok([string]$m)   { Write-Host "[+] $m" -ForegroundColor Green }
function Warn([string]$m) { Write-Host "[!] $m" -ForegroundColor Yellow }
function Fail([string]$m) { Write-Host "[x] $m" -ForegroundColor Red }

if (-not (Test-Path $ProjectPath)) { Fail "Project path not found: $ProjectPath"; exit 1 }
Set-Location $ProjectPath

# ---------- 1. git available ----------
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Fail "git is not installed. Get it from https://git-scm.com/download/win"
    exit 1
}
Info ("Using " + (git --version))

# ---------- 2. init ----------
if (-not (Test-Path ".git")) {
    Info "Initializing repository (default branch: main)"
    git init -b main | Out-Null
    Ok "Repository initialized"
} else {
    Info "Repository already exists, skipping init"
}

# ---------- 3. never commit this helper ----------
$excludeFile = Join-Path ".git" "info\exclude"
if (Test-Path $excludeFile) {
    $existing = @(Get-Content $excludeFile -ErrorAction SilentlyContinue)
    if (-not ($existing -contains "push_to_github.ps1")) {
        Add-Content -Path $excludeFile -Value "push_to_github.ps1"
    }
}

# ---------- 4. author identity ----------
$name  = git config user.name
$email = git config user.email
if (-not $name -or -not $email) {
    Warn "git identity is not configured. Run these two lines first, then re-run this script:"
    Write-Host '    git config --global user.name  "Your Name"'
    Write-Host '    git config --global user.email "you@example.com"'
    exit 1
}
Ok "Author: $name <$email>"

# ---------- 5. ignore rules must really work ----------
Info "Verifying .gitignore"
$critical = @(".env", ".venv", ".idea", "data/output", "test_milvus/milvus_demo.db")
$leaked = @()
foreach ($p in $critical) {
    if (Test-Path $p) {
        $r = git check-ignore $p
        if (-not $r) { $leaked += $p }
    }
}
if ($leaked.Count -gt 0) {
    Fail ("These paths are NOT ignored and would be committed: " + ($leaked -join ", "))
    Fail "Fix .gitignore before pushing."
    exit 1
}
Ok "Secrets / venv / IDE / logs / build output are excluded"

# ---------- 6. optional: keep PDFs local ----------
if ($ExcludePdf) {
    if (Test-Path $excludeFile) {
        Info "Excluding data/*.pdf via .git/info/exclude (local only)"
        Add-Content -Path $excludeFile -Value "`ndata/*.pdf`ndata/**/*.pdf"
    }
}

# ---------- 7. large file scan ----------
Info "Scanning for files larger than 50 MB (GitHub rejects anything above 100 MB)"
$skip = '\\\.git\\|\\\.venv\\|\\\.idea\\|\\__pycache__\\|\\logs\\'
$big = @(Get-ChildItem -Path $ProjectPath -Recurse -File -ErrorAction SilentlyContinue |
         Where-Object { $_.FullName -notmatch $skip -and $_.Length -gt 50MB })
if ($big.Count -gt 0) {
    foreach ($f in $big) {
        Warn ("{0,7:N1} MB  {1}" -f ($f.Length / 1MB), $f.FullName.Substring($ProjectPath.Length + 1))
    }
    Warn "Re-run with -ExcludePdf, or move these files out of the project before pushing."
} else {
    Ok "No oversized file"
}

# ---------- 8. stage + commit ----------
git add -A
$staged = @(git diff --cached --name-only)
Info ("Staged files: {0}" -f $staged.Count)

$bad = @($staged | Where-Object { $_ -match '^(\.venv/|\.idea/|\.env$|logs/|data/output/)' })
if ($bad.Count -gt 0) {
    Fail "Unexpected staged paths:"
    $bad | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
    Fail "Aborting before commit."
    exit 1
}

if ($staged.Count -eq 0) {
    Warn "Nothing to commit, working tree is clean"
} else {
    git commit -m "feat: RAG knowledge-base QA with hybrid retrieval and rerank" | Out-Null
    Ok "Commit created"
}

# ---------- 9. remote + push ----------
if ($RepoUrl) {
    $remotes = @(git remote)
    if ($remotes -contains "origin") {
        git remote set-url origin $RepoUrl
    } else {
        git remote add origin $RepoUrl
    }
    Ok "origin -> $RepoUrl"
}

$remotes = @(git remote)
if ($remotes.Count -eq 0) {
    Warn "No remote configured yet. Finish manually:"
    Write-Host "    git remote add origin https://github.com/<user>/<repo>.git"
    Write-Host "    git push -u origin main"
    exit 0
}

git branch -M main
Info "Pushing to GitHub ..."
git push -u origin main
if ($LASTEXITCODE -ne 0) { Fail "push failed, see the message above"; exit 1 }
Ok "Done: local repository pushed to GitHub"
