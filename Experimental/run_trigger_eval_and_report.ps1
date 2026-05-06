param(
    [string]$RunId = "",
    [string]$Models = "main",
    [int]$Concurrency = 200,
    [int]$RequestTimeout = 30,
    [int]$MaxAttempts = 12,
    [int]$MaxItems = 0,
    [string]$AdaptiveTriggerModel = "openai/gpt-5.4-mini",
    [string]$AdaptiveTriggerCheckerModel = "openai/gpt-5.4-mini",
    [switch]$UseRunIdPrefix,
    [switch]$DryRun,
    [switch]$SkipTemporal
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
$ExperimentalDir = Join-Path $RepoRoot "Experimental"

if (-not $RunId) {
    $RunId = "trigger_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

$ResultsDir = Join-Path $ExperimentalDir "results"
$ReportDir = Join-Path $ExperimentalDir ("reports\" + $RunId)
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null

$HeteroSequences = @(
    "authority,scarcity,unity",
    "reciprocity,liking,unity",
    "simple_baseline,consistency,social_proof",
    "social_proof,simple_baseline,consistency",
    "simple_baseline,authority,reciprocity",
    "consistency,scarcity,liking"
)

function Result-Path {
    param([string]$Stem)
    if ($UseRunIdPrefix) {
        return "results/${RunId}_${Stem}.jsonl.gz"
    }
    return "results/${Stem}.jsonl.gz"
}

function Add-CommonArgs {
    param([string[]]$CommandArgs)
    $CommandArgs = $CommandArgs + @(
        "--concurrency", [string]$Concurrency,
        "--request-timeout", [string]$RequestTimeout,
        "--max-attempts", [string]$MaxAttempts
    )
    if ($MaxItems -gt 0) {
        $CommandArgs = $CommandArgs + @("--max-items", [string]$MaxItems)
    }
    return $CommandArgs
}

function Add-AdaptiveArgs {
    param(
        [string[]]$CommandArgs,
        [string]$Mode
    )
    if ($Mode -eq "adaptive") {
        return $CommandArgs + @(
            "--adaptive-trigger-model", $AdaptiveTriggerModel,
            "--adaptive-trigger-checker-model", $AdaptiveTriggerCheckerModel
        )
    }
    return $CommandArgs
}

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$CommandArgs
    )
    Write-Host "==== $Name ===="
    if ($DryRun -and ($CommandArgs[0] -eq "Experimental\run.py")) {
        $CommandArgs = $CommandArgs + @("--dry-run")
    }
    & python @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

Push-Location $RepoRoot
try {
    Invoke-Step "trigger panel audit" @("Experimental\data\helper\audit_supersycophantic_panels.py")

    foreach ($Branch in @("gt", "ngt")) {
        $InputName = if ($Branch -eq "gt") {
            "data/supersycophantic_trigger_gt_neutral_200.jsonl"
        } else {
            "data/supersycophantic_trigger_ngt_neutral_100.jsonl"
        }
        $FirstTurn = Result-Path "${Branch}_first_turn"

        $FirstTurnArgs = Add-CommonArgs @(
            "Experimental\run.py", "first-turn",
            "--input", $InputName,
            "--models", $Models,
            "--output", $FirstTurn
        )
        Invoke-Step "$Branch first-turn cache" $FirstTurnArgs

        foreach ($Mode in @("static", "adaptive")) {
            $SingleArgs = Add-CommonArgs (Add-AdaptiveArgs @(
                "Experimental\run.py", "eval",
                "--input", $InputName,
                "--models", $Models,
                "--triggers", "all",
                "--tones", "mild", "moderate", "strong",
                "--trigger-prompt-mode", $Mode,
                "--initial-cache-from", $FirstTurn,
                "--output", (Result-Path "${Branch}_trigger_${Mode}")
            ) $Mode)
            Invoke-Step "$Branch $Mode single-trigger eval" $SingleArgs

            if (-not $SkipTemporal) {
                $TemporalOutput = Result-Path "${Branch}_trigger_temporal_${Mode}"

                $SameFamilyArgs = Add-CommonArgs (Add-AdaptiveArgs @(
                    "Experimental\run.py", "temporal",
                    "--input", $InputName,
                    "--models", $Models,
                    "--triggers", "all",
                    "--tone-sequence", "mild", "moderate", "strong",
                    "--trigger-prompt-mode", $Mode,
                    "--initial-cache-from", $FirstTurn,
                    "--output", $TemporalOutput
                ) $Mode)
                Invoke-Step "$Branch $Mode same-family temporal eval" $SameFamilyArgs

                $HeteroArgs = @(
                    "Experimental\run.py", "temporal",
                    "--input", $InputName,
                    "--models", $Models,
                    "--trigger-sequences"
                )
                $HeteroArgs = $HeteroArgs + $HeteroSequences + @(
                    "--tone-sequence", "mild", "moderate", "strong",
                    "--trigger-prompt-mode", $Mode,
                    "--initial-cache-from", $FirstTurn,
                    "--output", $TemporalOutput
                )
                $HeteroArgs = Add-CommonArgs (Add-AdaptiveArgs $HeteroArgs $Mode)
                Invoke-Step "$Branch $Mode hetero temporal eval" $HeteroArgs
            }
        }
    }

    $ReportArgs = @(
        "Experimental\report_results.py",
        "--results-dir", "Experimental\results",
        "--report-dir", ("Experimental\reports\" + $RunId)
    )
    if ($UseRunIdPrefix) {
        $ReportArgs = $ReportArgs + @("--run-id", $RunId)
    }
    Invoke-Step "trigger report" $ReportArgs
    Invoke-Step "trigger figure candidates" @(
        "Experimental\plot_trigger_figures.py",
        "--run-id", $RunId,
        "--results-dir", "Experimental\results",
        "--report-dir", ("Experimental\reports\" + $RunId),
        "--clean"
    )
    Write-Host "RunId=$RunId"
    Write-Host ("Report=" + (Join-Path $ReportDir "report.md"))
    Write-Host ("Figures=" + (Join-Path $ReportDir "paper_figure_candidates"))
}
finally {
    Pop-Location
}
