param(
    [string]$RunId = "",
    [string]$Models = "main",
    [int]$Concurrency = 200,
    [int]$RequestTimeout = 30,
    [int]$MaxAttempts = 12,
    [int]$MaxGt = 0,
    [int]$MaxNgt = 0,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
$ExperimentalDir = Join-Path $RepoRoot "Experimental"

if (-not $RunId) {
    $RunId = "context_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

$ResultsDir = Join-Path $ExperimentalDir "results"
$ReportDir = Join-Path $ExperimentalDir ("reports\" + $RunId)
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$CommandArgs
    )
    Write-Host "==== $Name ===="
    if ($DryRun -and ($CommandArgs[0] -eq "Experimental\run_context.py")) {
        $CommandArgs = $CommandArgs + @("--dry-run")
    }
    & python @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

Push-Location $RepoRoot
try {
    Invoke-Step "build context panels" @(
        "Experimental\data\helper\build_supersycophantic_context_panels.py",
        "--write",
        "--audit", "Experimental\data\context_source_traceability_audit.md"
    )
    Invoke-Step "panel audit" @("Experimental\data\helper\audit_supersycophantic_panels.py")
    Invoke-Step "context panel integrity audit" @(
        "Experimental\data\helper\audit_context_panel_integrity.py",
        "--report", "Experimental\data\context_panel_integrity_audit.md"
    )
    Invoke-Step "framing naturalness audit" @(
        "Experimental\data\helper\audit_context_framing_naturalness.py",
        "--report", "Experimental\data\context_framing_naturalness_audit.md"
    )

    $ContextArgs = @(
        "Experimental\run_context.py",
        "--models", $Models,
        "--output", "results/${RunId}_context_main.jsonl.gz",
        "--summary", "results/${RunId}_context_main_summary.json",
        "--concurrency", [string]$Concurrency,
        "--request-timeout", [string]$RequestTimeout,
        "--max-attempts", [string]$MaxAttempts
    )
    if ($MaxGt -gt 0) {
        $ContextArgs = $ContextArgs + @("--max-gt", [string]$MaxGt)
    }
    if ($MaxNgt -gt 0) {
        $ContextArgs = $ContextArgs + @("--max-ngt", [string]$MaxNgt)
    }
    Invoke-Step "context eval" $ContextArgs

    if ($DryRun) {
        Write-Host "DryRun complete; skipped report generation because no result summary is written."
        return
    }

    Invoke-Step "context report" @(
        "Experimental\report_results.py",
        "--results-dir", "Experimental\results",
        "--report-dir", ("Experimental\reports\" + $RunId),
        "--run-id", $RunId,
        "--include-context"
    )
    Write-Host "RunId=$RunId"
    Write-Host ("Report=" + (Join-Path $ReportDir "report.md"))
}
finally {
    Pop-Location
}
