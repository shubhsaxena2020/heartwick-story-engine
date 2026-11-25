# Registers (or updates) the scheduled task "Heartwick Studio": at logon + 2 min, then every 15 min, it runs
# tools\start_heartwick.ps1 hidden (conhost.exe --headless), which starts only what is not already running.
#   powershell -ExecutionPolicy Bypass -File tools\register_heartwick_task.ps1
$root = Split-Path -Parent $PSScriptRoot
$script = Join-Path $root "tools\start_heartwick.ps1"
$action = New-ScheduledTaskAction -Execute "conhost.exe" `
    -Argument "--headless powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$script`"" -WorkingDirectory $root
$logon = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$logon.Delay = "PT2M"
$every = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 15)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName "Heartwick Studio" -Action $action -Trigger @($logon, $every) -Settings $settings `
    -Principal $principal -Description "Heartwick Studio dashboard: exporter, live channel, queue runner (only what is not running)" -Force |
    Select-Object TaskName, State
