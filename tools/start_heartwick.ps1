# Bring Heartwick Studio's laptop side up (after a reboot, or any time): each step starts only what is not running.
#   powershell -ExecutionPolicy Bypass -File tools\start_heartwick.ps1
# Registered as the scheduled task "Heartwick Studio" (logon + 2 min, then every 15 min), run hidden through
# conhost.exe --headless. See tools\register_heartwick_task.ps1.
#
#  1. dashboard\exporter.py  snapshot + media -> Vercel (heartwick-studio), queued approvals -> approvals.json
#  2. dashboard\live.py      127.0.0.1:8788 + cloudflared quick tunnel for the site's live view
#  3. tools\run_queue.py     only when kaggle\queue.json has a pending job and no runner is alive
# CPU rule: main laptop <= 75 %. 1 and 2 run BelowNormal on CPUs 12-13 only (the exporter's 480p ffmpeg inherits that);
# the queue runner only waits on Kaggle. Nothing here is ever stopped by image name.

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$py = "C:\Program Files\Python313\python.exe"
$logs = Join-Path $root "dashboard\data\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
$log = Join-Path $logs ("start_" + (Get-Date -Format "yyyyMMdd") + ".log")
function Log($m) { $line = "[" + (Get-Date -Format "HH:mm:ss") + "] " + $m; Add-Content -Path $log -Value $line; Write-Output $line }
$MASK = [IntPtr]0x3000          # CPUs 12-13

function Running($pattern) {
    @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match $pattern }).Count -gt 0
}

function Start-Part($name, $pattern, $script, $argList, $outName, $mask) {
    if (Running $pattern) { Log "$name already running"; return }
    $out = Join-Path $logs $outName
    foreach ($f in $out, ($out + ".err")) { if (Test-Path $f) { Move-Item -Force $f ($f + ".prev") -ErrorAction SilentlyContinue } }
    $p = Start-Process -FilePath $py -ArgumentList (@("-u", "-X", "utf8", (Join-Path $root $script)) + $argList) `
        -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput $out -RedirectStandardError ($out + ".err")
    Start-Sleep 1
    try { $p.PriorityClass = "BelowNormal"; if ($mask) { $p.ProcessorAffinity = $mask } } catch {}
    Log "$name started (pid $($p.Id))"
}

Start-Part "exporter" "dashboard[\\/]exporter\.py" "dashboard\exporter.py" @("--loop-seconds", "60") "exporter.out" $MASK
Start-Part "live channel" "dashboard[\\/]live\.py" "dashboard\live.py" @() "live.out" $MASK

# queue runner: only with work to do (a restart must never re-run finished Kaggle jobs)
$q = Join-Path $root "kaggle\queue.json"
$pending = 0
if (Test-Path $q) { try { $pending = @((Get-Content $q -Raw | ConvertFrom-Json).jobs | Where-Object { $_.status -eq "pending" }).Count } catch {} }
if (Running "run_queue\.py") { Log "queue runner already running" }
elseif ($pending -gt 0) { Start-Part "queue runner ($pending pending)" "run_queue\.py" "tools\run_queue.py" @() "run_queue.out" $null }
else { Log "queue runner: nothing pending" }
Log "start_heartwick done"
