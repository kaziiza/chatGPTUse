param(
    [string[]]$Symbols = @("BTCUSDT", "ETHUSDT", "SOLUSDT", "HYPEUSDT"),
    [int]$CaptureDurationSec = 172800,
    [int]$ReplayDurationSec = 61200,
    [double]$Capital = 10000,
    [double]$MaxLeverage = 5,
    [double]$MaxOrderNotional = 400,
    [double]$MinOrderNotional = 10,
    [string]$RestStart = "",
    [string]$RestEnd = "",
    [switch]$SkipRestBackfill
)

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$wsOut = Join-Path $ProjectRoot "data/raw/bybit_ws"
$restOut = Join-Path $ProjectRoot "data/backfill/bybit"
$mergedRoot = Join-Path $ProjectRoot "data/tsmixer/datasets/merged_$timestamp"
$csvRoot = Join-Path $ProjectRoot "data/tsmixer/datasets/merged_csv_$timestamp"
$datasetOut = Join-Path $ProjectRoot "data/datasets/tsmixer_$timestamp"
$ts2vecOut = Join-Path $ProjectRoot "data/embeddings/ts2vec_$timestamp"
$ts2vecModelOut = Join-Path $ProjectRoot "models/ts2vec_$timestamp"
$tsmixerPtStamp = Join-Path $ProjectRoot "data/models/tsmixer_poc_$timestamp.pt"
$tsmixerStatsStamp = Join-Path $ProjectRoot "data/models/tsmixer_poc_$timestamp.stats.json"
$tsmixerMetaStamp = Join-Path $ProjectRoot "data/models/tsmixer_poc_$timestamp.meta.json"
$tsmixerPtCanonical = Join-Path $ProjectRoot "data/models/tsmixer_poc.pt"
$tsmixerStatsCanonical = Join-Path $ProjectRoot "data/models/tsmixer_poc.stats.json"
$tsmixerMetaCanonical = Join-Path $ProjectRoot "data/models/tsmixer_poc.meta.json"

function Invoke-PyStep {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
    Write-Host "`n[$(Get-Date -Format o)] >>> $Name" -ForegroundColor Cyan
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Step '$Name' failed with exit code $LASTEXITCODE"
    }
}

function Get-LatestReplayFile {
    param([string]$Symbol)
    $symRoot = Join-Path $wsOut $Symbol
    if (-not (Test-Path $symRoot)) {
        throw "Replay directory not found for $Symbol at $symRoot"
    }
    $latest = Get-ChildItem -Path $symRoot -Recurse -File -Include *.jsonl,*.jsonl.gz |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $latest) {
        throw "No replay dump found for $Symbol under $symRoot"
    }
    return $latest.FullName
}

function Convert-MergedToCsv {
    param(
        [string]$InputRoot,
        [string]$OutputRoot,
        [string[]]$Symbols
    )
    $symbolsJson = ($Symbols | ConvertTo-Json -Compress)
    $script = @"
import json
import pandas as pd
from pathlib import Path

symbols = json.loads('$symbolsJson')
input_root = Path(r'$InputRoot')
output_root = Path(r'$OutputRoot')
output_root.mkdir(parents=True, exist_ok=True)

for sym in symbols:
    parquet_path = input_root / sym / f"{sym}.parquet"
    if not parquet_path.exists():
        continue
    df = pd.read_parquet(parquet_path)
    if 'ts' not in df.columns:
        raise RuntimeError(f"'ts' column missing in {parquet_path}")
    df['ts'] = pd.to_datetime(df['ts'], utc=True)
    df = df.sort_values('ts')
    for day, chunk in df.groupby(df['ts'].dt.strftime('%Y-%m-%d')):
        out_dir = output_root / sym
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{day}.csv"
        chunk.to_csv(out_path, index=False)
"@
    $script | & python -
    if ($LASTEXITCODE -ne 0) {
        throw "CSV conversion failed"
    }
}

$symbolsCsv = ($Symbols -join ",")

# 1) 48h websocket capture
$collectArgs = @("scripts/collect_bybit_ws.py", "--symbols") + $Symbols + @(
    "--duration", $CaptureDurationSec.ToString(),
    "--out", $wsOut,
    "--compress"
)
Invoke-PyStep "Collect Bybit WS dumps" $collectArgs

# 2) Replay each symbol into feature dumps
foreach ($symbol in $Symbols) {
    $replayFile = Get-LatestReplayFile -Symbol $symbol
    $reportDir = Join-Path $ProjectRoot "runs/replay_feature_dump/$symbol"
    New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
    $reportPath = Join-Path $reportDir "report_$timestamp.json"
    $runnerArgs = @(
        "src/runner/perp_paper_runner.py",
        "--symbol", $symbol,
        "--duration", $ReplayDurationSec.ToString(),
        "--capital", [string]$Capital,
        "--max-leverage", [string]$MaxLeverage,
        "--max-order-notional", [string]$MaxOrderNotional,
        "--min-order-notional", [string]$MinOrderNotional,
        "--maker-fee-bps", "0.02",
        "--taker-fee-bps", "0.055",
        "--report-path", $reportPath,
        "--log-dir", $reportDir,
        "--overlays", "mark_sanitize", "latency_guard", "pr047_feature_dump",
        "--ws-replay", $replayFile,
        "--ws-replay-speed", "240"
    )
    Invoke-PyStep "Replay $symbol into feature dumps" $runnerArgs
}

# 3) Optional REST backfill
if (-not $SkipRestBackfill.IsPresent -and $RestStart -and $RestEnd) {
    $restArgs = @(
        "scripts/pull_bybit_history.py",
        "--symbols", $symbolsCsv,
        "--start", $RestStart,
        "--end", $RestEnd,
        "--out", $restOut
    )
    Invoke-PyStep "Pull Bybit REST history" $restArgs
} else {
    Write-Host "Skipping REST backfill (no window supplied or --SkipRestBackfill set)." -ForegroundColor Yellow
}

# 4) Merge WS + REST into patched dataset
$patchArgs = @(
    "scripts/patch_dataset.py",
    "--src-ws", "data/tsmixer/datasets/poc",
    "--src-rest", "data/backfill/bybit",
    "--out-root", $mergedRoot,
    "--symbols", $symbolsCsv,
    "--resample-sec", "10",
    "--rest-disallow-micro", "true",
    "--require-complete", "mark_price,index_price,mid"
)
Invoke-PyStep "Merge WS/REST feature tables" $patchArgs

# 5) Build supervised dataset for TSMixer
$buildArgs = @(
    "scripts/build_tsmixer_dataset.py",
    "--data-root", $mergedRoot,
    "--symbols", $symbolsCsv,
    "--resample-sec", "10",
    "--seq-len", "600",
    "--horizon-sec", "900",
    "--cost-bps", "30",
    "--winsorize-ofi", "5",
    "--time-ordered",
    "--add-view-id",
    "--out", $datasetOut
)
Invoke-PyStep "Build TSMixer dataset" $buildArgs

# 6) Convert merged parquet shards into daily CSVs for TS2Vec
Convert-MergedToCsv -InputRoot $mergedRoot -OutputRoot $csvRoot -Symbols $Symbols

# 7) Train TS2Vec embeddings
$ts2vecArgs = @(
    "scripts/train_ts2vec.py",
    "--symbols", $symbolsCsv,
    "--data-root", $csvRoot,
    "--window", "600",
    "--train-stride", "10",
    "--export-stride", "5",
    "--epochs", "200",
    "--batch-size", "512",
    "--hidden-dim", "256",
    "--depth", "4",
    "--emb-dim", "128",
    "--proj-dim", "128",
    "--dropout", "0.1",
    "--tau", "0.2",
    "--lr", "0.001",
    "--device", "cuda",
    "--out-root", $ts2vecOut,
    "--model-out", $ts2vecModelOut
)
Invoke-PyStep "Train TS2Vec embeddings" $ts2vecArgs

# 8) Train / export new TSMixer weights
$tsmixerArgs = @(
    "scripts/train_tsmixer_poc.py",
    "--dataset", $datasetOut,
    "--seq-len", "600",
    "--horizon-sec", "900",
    "--batch-size", "256",
    "--epochs", "30",
    "--lr", "0.001",
    "--weight-decay", "0.0001",
    "--dropout", "0.1",
    "--device", "cuda",
    "--save-pt", $tsmixerPtStamp,
    "--save-stats", $tsmixerStatsStamp,
    "--save-meta", $tsmixerMetaStamp
)
Invoke-PyStep "Train TSMixer PoC model" $tsmixerArgs

Copy-Item -Force $tsmixerPtStamp $tsmixerPtCanonical
Copy-Item -Force $tsmixerStatsStamp $tsmixerStatsCanonical
Copy-Item -Force $tsmixerMetaStamp $tsmixerMetaCanonical

Write-Host "`nPipeline finished. Artefacts:" -ForegroundColor Green
Write-Host "  Patched dataset : $mergedRoot"
Write-Host "  TS2Vec CSV root : $csvRoot"
Write-Host "  TS2Vec embeddings: $ts2vecOut"
Write-Host "  TSMixer dataset : $datasetOut"
Write-Host "  TSMixer weights : $tsmixerPtCanonical"
