"""Exercise Pause/Resume with a fake Azure CLI; never contact Azure."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest


def test_pause_resume_finds_inactive_revisions_and_preserves_order(tmp_path):
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if not shell:
        pytest.skip("PowerShell not installed")
    script = Path(__file__).resolve().parents[1] / "scripts" / "Manage-RagDemo.ps1"
    harness = tmp_path / "lifecycle.ps1"
    state = tmp_path / "state.json"
    operations = tmp_path / "operations.json"
    code = r'''
param($ScriptPath, $StatePath, $OperationsPath)
$ErrorActionPreference = 'Stop'
$global:active = @{'ca-it-rag-api'=$true; 'ca-it-rag-ui'=$true}
$global:ops = @()
$global:suffix = 'saved'
function global:az {
    $parts = @($args)
    $global:LASTEXITCODE = 0
    $name = $parts[[Array]::IndexOf($parts, '--name') + 1]
    $revisionName = "$name--$global:suffix"
    if ($parts[1] -eq 'show') {
        return (@{properties=@{
            latestRevisionName=$revisionName
            configuration=@{activeRevisionsMode='Single';ingress=@{fqdn='localhost.invalid'}}
            template=@{containers=@(@{image='local:test'})}
        }} | ConvertTo-Json -Depth 10 -Compress)
    }
    if ($parts[1] -ne 'revision') { throw 'Unexpected command' }
    if ($parts[2] -eq 'deactivate') {
        $global:active[$name] = $false
        $global:ops += "stop:$name"
        return '{}'
    }
    if ($parts[2] -eq 'activate') {
        $global:active[$name] = $true
        $global:ops += "start:$name"
        return '{}'
    }
    $revision = @{name=$revisionName;properties=@{
        active=$global:active[$name]; healthState='Healthy'
        template=@{containers=@(@{image='local:test'})}
    }}
    if ($parts[2] -eq 'list') {
        # Match CLI semantics: inactive revisions are hidden unless --all is used.
        if (-not $global:active[$name] -and '--all' -notin $parts) { return '[]' }
        return (ConvertTo-Json -InputObject @($revision) -Depth 10 -Compress)
    }
    if ($parts[2] -eq 'show') { return ($revision | ConvertTo-Json -Depth 10 -Compress) }
    throw 'Unexpected revision command'
}
function global:Invoke-WebRequest { return @{StatusCode=200} }
& $ScriptPath -Action Pause -StatePath $StatePath
if ($global:active['ca-it-rag-api'] -or $global:active['ca-it-rag-ui']) { throw 'Pause failed' }
& $ScriptPath -Action Resume -StatePath $StatePath
if (-not $global:active['ca-it-rag-api'] -or -not $global:active['ca-it-rag-ui']) { throw 'Resume failed' }
ConvertTo-Json -InputObject $global:ops | Set-Content -LiteralPath $OperationsPath
# Simulate a manual restart and subsequent deployment, with stale paused state.
$stale = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
$stale.paused = $true
$stale | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $StatePath
$global:suffix = 'new'
$blocked = $false
try { & $ScriptPath -Action Resume -StatePath $StatePath } catch { $blocked = $true }
if (-not $blocked) { throw 'Stale Resume was not blocked' }
& $ScriptPath -Action Pause -StatePath $StatePath
& $ScriptPath -Action Resume -StatePath $StatePath
$newState = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
if ($newState.apps[0].revision -ne 'ca-it-rag-api--new') { throw 'Fresh Pause did not replace stale snapshot' }
'''
    harness.write_text(code, encoding="utf-8")
    result = subprocess.run([shell, "-NoProfile", "-File", str(harness),
                             str(script), str(state), str(operations)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(operations.read_text(encoding="utf-8-sig")) == [
        "stop:ca-it-rag-ui", "stop:ca-it-rag-api", "start:ca-it-rag-api", "start:ca-it-rag-ui"]
    assert not json.loads(state.read_text(encoding="utf-8-sig"))["paused"]
