param(
    [string[]]$Symbols = @("BTCUSDT","ETHUSDT","SOLUSDT","HYPEUSDT"),
    [string]$RawRoot = "data/raw/bybit_ws_20251201_72h",
    [string]$Ts = "20251201_72h_x20",
    [int]$ReplayDurationSec = 259200,
    [double]$ReplaySpeed = 20,
    [int]$SeqLen = 600,
    [int]$HorizonSec = 900
)

$ProjectRoot = (Get-Location).Path
$py = "C:\Users\user\anaconda3\python.exe"
Set-Location $ProjectRoot

function Latest-WsFile {
    param([string]$Sym)
    $files = Get-ChildItem -Path (Join-Path $RawRoot $Sym) -Recurse -File -Include *.jsonl,*.jsonl.gz |
        Sort-Object LastWriteTime -Descending
    if (-not $files) { throw "No WS file for $Sym in $RawRoot" }
    return $files[0].FullName
}

function Run-Step {
    param([string]$Name,[string[]]$CmdArgs)
    Write-Host "[`$(Get-Date -Format o)] $Name" -ForegroundColor Cyan
    & $py @CmdArgs
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with $LASTEXITCODE" }
}

# 1) Replay -> feature dump (sequential)
foreach ($sym in $Symbols) {
    $ws = Latest-WsFile -Sym $sym
    $reportDir = "runs/replay_feature_dump_${Ts}/$sym"
    New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
    $reportPath = Join-Path $reportDir "report_${Ts}.json"
    $args = @(
        "src/runner/perp_paper_runner.py",
        "--symbol", $sym,
        "--duration", [int]($ReplayDurationSec / [math]::Max($ReplaySpeed,1)) ,
        "--capital", "10000",
        "--max-leverage", "5",
        "--report-path", $reportPath,
        "--log-dir", $reportDir,
        "--overlays", "mark_sanitize", "latency_guard", "pr047_feature_dump_20251201",
        "--ws-replay", $ws,
        "--ws-replay-speed", $ReplaySpeed.ToString()
    )
    Run-Step "Replay $sym" $args
}

# 2) Patch dataset (WS only)
$mergedRoot = "data/tsmixer/datasets/merged_${Ts}"
$csvRoot = "data/tsmixer/datasets/merged_csv_${Ts}"
$symbolsCsv = ($Symbols -join ',')
Run-Step "Patch dataset" @(
    "scripts/patch_dataset.py",
    "--src-ws", "data/tsmixer/datasets/poc_20251201_72h",
    "--src-rest", "data/backfill/bybit",
    "--out-root", $mergedRoot,
    "--symbols", $symbolsCsv,
    "--resample-sec", "10",
    "--rest-disallow-micro", "true",
    "--require-complete", "mark_price,index_price,mid"
)

# 3) Build TSMixer datasets per symbol
foreach ($sym in $Symbols) {
    $out = "data/datasets/tsmixer_${sym}_${Ts}"
    Run-Step "Build dataset $sym" @(
        "scripts/build_tsmixer_dataset.py",
        "--data-root", $mergedRoot,
        "--symbols", $sym,
        "--resample-sec", "10",
        "--seq-len", $SeqLen.ToString(),
        "--horizon-sec", $HorizonSec.ToString(),
        "--cost-bps", "30",
        "--winsorize-ofi", "5",
        "--time-ordered",
        "--add-view-id",
        "--out", $out
    )
}

# 4) Convert merged parquet -> CSV for TS2Vec
Run-Step "Merged->CSV" @(
    "scripts/patch_dataset.py",
    "--src-ws", $mergedRoot,
    "--src-rest", "data/backfill/bybit",
    "--out-root", $csvRoot,
    "--symbols", $symbolsCsv,
    "--resample-sec", "10",
    "--rest-disallow-micro", "true",
    "--require-complete", "mark_price,index_price,mid"
)

# 5) TS2Vec per symbol (ETH設定: ep75, bs256)
foreach ($sym in $Symbols) {
    Run-Step "TS2Vec $sym" @(
        "scripts/train_ts2vec.py",
        "--symbols", $sym,
        "--data-root", $csvRoot,
        "--window", "600",
        "--train-stride", "10",
        "--export-stride", "5",
        "--epochs", "75",
        "--batch-size", "256",
        "--hidden-dim", "256",
        "--depth", "4",
        "--emb-dim", "128",
        "--proj-dim", "128",
        "--dropout", "0.1",
        "--tau", "0.2",
        "--lr", "0.001",
        "--device", "cuda",
        "--out-root", "data/embeddings/ts2vec_${Ts}",
        "--model-out", "models/ts2vec_${Ts}"
    )
}

# 6) TSMixer per symbol (ETH設定: ep75, bs256)
foreach ($sym in $Symbols) {
    $ds = "data/datasets/tsmixer_${sym}_${Ts}"
    Run-Step "TSMixer $sym" @(
        "scripts/train_tsmixer_poc.py",
        "--dataset", $ds,
        "--seq-len", $SeqLen.ToString(),
        "--batch-size", "256",
        "--epochs", "75",
        "--lr", "0.001",
        "--weight-decay", "0.0001",
        "--dropout", "0.1",
        "--device", "cuda",
        "--save-pt", "data/models/tsmixer_poc_${sym}_${Ts}_ep75.pt",
        "--save-stats", "data/models/tsmixer_poc_${sym}_${Ts}_ep75.stats.json",
        "--save-meta", "data/models/tsmixer_poc_${sym}_${Ts}_ep75.meta.json"
    )
}
