#Requires -Version 5.1
<#
.SYNOPSIS
Pause or resume the existing RAG Container Apps without deleting configuration.
.EXAMPLE
.\scripts\Manage-RagDemo.ps1 -Action Pause -WhatIf
.EXAMPLE
.\scripts\Manage-RagDemo.ps1 -Action Pause
.EXAMPLE
.\scripts\Manage-RagDemo.ps1 -Action Resume
#>
[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Medium')]
param(
    [ValidateSet('Status', 'Pause', 'Resume')]
    [string]$Action = 'Status',
    [string]$ResourceGroup = 'rag-demo-rg',
    [string]$SubscriptionId = '5a6ffaf0-ef35-40f8-a057-3172188652e1',
    [string]$StatePath = (Join-Path $PSScriptRoot '../.local/rag-demo/state.json'),
    [ValidateRange(60, 1800)]
    [int]$StartupTimeoutSeconds = 600
)

$ErrorActionPreference = 'Stop'
$appNames = @('ca-it-rag-api', 'ca-it-rag-ui') # Startup order; shutdown reverses it.
if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    throw 'Azure CLI is required. Install it, run az login, then retry.'
}

function Invoke-AzureJson {
    param([string[]]$CliArguments)
    $result = & az @CliArguments --subscription $SubscriptionId --only-show-errors --output json
    if ($LASTEXITCODE -ne 0) {
        throw "Azure CLI failed: az $($CliArguments -join ' '). Check az login and permissions."
    }
    if ($result) { return (($result -join "`n") | ConvertFrom-Json) }
}

function Get-DemoApp {
    param([string]$AppName)
    Invoke-AzureJson @('containerapp', 'show', '--name', $AppName, '--resource-group', $ResourceGroup)
}

function Get-DemoRevisions {
    param([string]$AppName)
    Invoke-AzureJson @('containerapp', 'revision', 'list', '--all', '--name', $AppName, '--resource-group', $ResourceGroup)
}

function Save-DemoState {
    param($DemoState)
    $directory = Split-Path -Parent $StatePath
    if ($directory -and -not (Test-Path -LiteralPath $directory)) {
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
    }
    # Metadata only: never export secrets, tokens or environment variable values.
    $DemoState | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $StatePath -Encoding UTF8
}

$apps = @{}
foreach ($appName in $appNames) { $apps[$appName] = Get-DemoApp $appName }

if ($Action -eq 'Status') {
    foreach ($appName in $appNames) {
        $app = $apps[$appName]
        $active = @(Get-DemoRevisions $appName | Where-Object { $_.properties.active })
        [pscustomobject]@{
            App = $appName
            Image = $app.properties.template.containers[0].image
            ActiveRevisions = $active.Count
            Revision = ($active.name -join ', ')
            URL = "https://$($app.properties.configuration.ingress.fqdn)"
        }
    }
    return
}

$state = $null
if (Test-Path -LiteralPath $StatePath) {
    $state = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
    if ($state.subscriptionId -ne $SubscriptionId -or $state.resourceGroup -ne $ResourceGroup) {
        throw 'Saved state belongs to another subscription/resource group. Use the matching parameters.'
    }
    if ((@($state.apps.name | Sort-Object) -join ',') -ne (@($appNames | Sort-Object) -join ',')) {
        throw 'Saved state does not contain the expected API and UI apps.'
    }
}

if ($Action -eq 'Pause') {
    $snapshotOutdated = $false
    if ($state) {
        $snapshotOutdated = @($state.apps | Where-Object {
            $apps[$_.name].properties.latestRevisionName -ne $_.latestRevision
        }).Count -gt 0
    }
    # A later deployment may follow a manual portal restart. A fresh Pause can
    # capture its active revisions; Resume must still reject stale snapshots.
    if (-not $state -or -not $state.paused -or $snapshotOutdated) {
        $snapshots = foreach ($appName in $appNames) {
            $app = $apps[$appName]
            if ($app.properties.configuration.activeRevisionsMode -ne 'Single') {
                throw "$appName is not in Single revision mode. Review traffic routing before pausing."
            }
            $active = @(Get-DemoRevisions $appName | Where-Object { $_.properties.active })
            if ($active.Count -ne 1) {
                throw "$appName must have exactly one active revision to save a restart point."
            }
            [pscustomobject]@{
                name = $appName
                revision = $active[0].name
                latestRevision = $app.properties.latestRevisionName
                image = $active[0].properties.template.containers[0].image
            }
        }
        $state = [pscustomobject]@{
            version = 1
            subscriptionId = $SubscriptionId
            resourceGroup = $ResourceGroup
            capturedUtc = [DateTime]::UtcNow.ToString('o')
            paused = $true
            apps = @($snapshots)
        }
        # Save before deactivation so Resume also works after a partial failure.
        if (-not $WhatIfPreference) { Save-DemoState $state }
    }
} elseif (-not $state) {
    throw "No saved restart point at $StatePath. Run Pause first; do not guess an old revision."
}

# Fail before any Azure change if a pipeline deployed a different revision.
foreach ($saved in $state.apps) {
    if ($apps[$saved.name].properties.latestRevisionName -ne $saved.latestRevision) {
        throw "A newer revision exists for $($saved.name). Check pipeline activity before resuming saved revisions."
    }
    $available = @(Get-DemoRevisions $saved.name)
    if ($saved.revision -notin @($available.name)) {
        throw "Saved revision $($saved.revision) no longer exists. Redeploy through Azure DevOps."
    }
}

if ($Action -eq 'Pause') {
    foreach ($appName in @('ca-it-rag-ui', 'ca-it-rag-api')) {
        $saved = $state.apps | Where-Object { $_.name -eq $appName }
        $active = @(Get-DemoRevisions $appName | Where-Object { $_.properties.active })
        if (@($active | Where-Object { $_.name -ne $saved.revision }).Count -gt 0) {
            throw "Unexpected active revision on $appName; inspect pipeline activity."
        }
        if ($active.Count -gt 0 -and $PSCmdlet.ShouldProcess($saved.revision, 'Deactivate revision and stop replicas')) {
            Invoke-AzureJson @('containerapp', 'revision', 'deactivate', '--name', $appName,
                '--resource-group', $ResourceGroup, '--revision', $saved.revision) | Out-Null
        }
    }
    if (-not $WhatIfPreference) {
        foreach ($appName in $appNames) {
            if (@(Get-DemoRevisions $appName | Where-Object { $_.properties.active }).Count -gt 0) {
                throw "$appName still has active revisions. Check Status before assuming compute is stopped."
            }
        }
        Write-Host 'Both apps have no active revisions. Replica shutdown may take a short time.'
        Write-Host "Restart metadata saved to $StatePath. Keep this file until Resume succeeds."
    }
} else {
    foreach ($appName in $appNames) {
        $saved = $state.apps | Where-Object { $_.name -eq $appName }
        $revision = Invoke-AzureJson @('containerapp', 'revision', 'show', '--name', $appName,
            '--resource-group', $ResourceGroup, '--revision', $saved.revision)
        if (-not $revision.properties.active -and $PSCmdlet.ShouldProcess($saved.revision, 'Activate saved revision')) {
            Invoke-AzureJson @('containerapp', 'revision', 'activate', '--name', $appName,
                '--resource-group', $ResourceGroup, '--revision', $saved.revision) | Out-Null
        }
        if ($WhatIfPreference) { continue }
        # The API has minReplicas=1. Wait for its replica before starting the UI.
        if ($appName -eq 'ca-it-rag-api') {
            $deadline = [DateTime]::UtcNow.AddSeconds($StartupTimeoutSeconds)
            do {
                $revision = Invoke-AzureJson @('containerapp', 'revision', 'show', '--name', $appName,
                    '--resource-group', $ResourceGroup, '--revision', $saved.revision)
                if ($revision.properties.active -and $revision.properties.healthState -eq 'Healthy') { break }
                if ([DateTime]::UtcNow -gt $deadline) { throw 'API did not become healthy before the startup timeout. Inspect its logs.' }
                Start-Sleep -Seconds 5
            } while ($true)
        }
    }
    if (-not $WhatIfPreference) {
        $uiUrl = "https://$($apps['ca-it-rag-ui'].properties.configuration.ingress.fqdn)"
        $deadline = [DateTime]::UtcNow.AddSeconds($StartupTimeoutSeconds)
        do {
            try {
                $response = Invoke-WebRequest -UseBasicParsing -Uri "$uiUrl/_stcore/health" -TimeoutSec 20
                if ($response.StatusCode -eq 200) { break }
            } catch { Write-Verbose $_.Exception.Message }
            if ([DateTime]::UtcNow -gt $deadline) { throw 'UI health check timed out. Inspect its revision and logs.' }
            Start-Sleep -Seconds 5
        } while ($true)
        $state.paused = $false
        Save-DemoState $state
        Write-Host "API revision is healthy; Streamlit health check passed: $uiUrl"
        Write-Host 'Open the UI and ask the VPN example to verify knowledge-base readiness and Azure OpenAI.'
    }
}
