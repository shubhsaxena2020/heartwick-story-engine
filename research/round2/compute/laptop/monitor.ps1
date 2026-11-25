# Samples CPU %, thermal-zone temperature and iGPU engine utilisation every $Every seconds, $Count times (bounded), to CSV.
param([string]$Out, [int]$Every = 15, [int]$Count = 480)
"time,cpu_pct,temp_c,gpu_compute_pct,gpu_3d_pct" | Out-File -Encoding utf8 $Out
for ($i = 0; $i -lt $Count; $i++) {
    $c = Get-Counter '\Processor(_Total)\% Processor Time', '\Thermal Zone Information(*)\Temperature', '\GPU Engine(*)\Utilization Percentage' -ErrorAction SilentlyContinue
    $s = $c.CounterSamples
    $cpu = ($s | Where-Object { $_.Path -like '*processor(_total)*' }).CookedValue
    $t = ($s | Where-Object { $_.Path -like '*thermal zone*' } | Measure-Object CookedValue -Maximum).Maximum - 273.15
    $gc = ($s | Where-Object { $_.Path -like '*engtype_compute*' } | Measure-Object CookedValue -Sum).Sum
    $g3 = ($s | Where-Object { $_.Path -like '*engtype_3d*' } | Measure-Object CookedValue -Sum).Sum
    "{0},{1:N0},{2:N1},{3:N0},{4:N0}" -f (Get-Date -Format 'HH:mm:ss'), $cpu, $t, $gc, $g3 | Out-File -Append -Encoding utf8 $Out
    Start-Sleep -Seconds $Every
}
