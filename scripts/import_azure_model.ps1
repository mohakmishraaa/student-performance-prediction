param(
    [Parameter(Mandatory = $true)] [string] $JobName,
    [Parameter(Mandatory = $true)] [string] $ResourceGroup,
    [Parameter(Mandatory = $true)] [string] $WorkspaceName
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$outputsPath = Join-Path $projectRoot 'outputs'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$downloadPath = Join-Path $outputsPath "azure-job-download-$stamp"
New-Item -ItemType Directory -Force -Path $outputsPath | Out-Null
New-Item -ItemType Directory -Force -Path $downloadPath | Out-Null

Write-Host "Downloading model_output from Azure ML job '$JobName'..."
az ml job download --name $JobName --output-name model_output --download-path $downloadPath `
    --resource-group $ResourceGroup --workspace-name $WorkspaceName
if ($LASTEXITCODE -ne 0) {
    throw 'Azure ML download failed. Check that the job completed and your account can access this workspace.'
}

$modelFile = Get-ChildItem -LiteralPath $downloadPath -Filter 'model.joblib' -File -Recurse |
    Select-Object -First 1
if (-not $modelFile) {
    throw "The downloaded output did not contain model.joblib. Downloaded files are in $downloadPath"
}
$artifactDirectory = $modelFile.Directory.FullName
foreach ($requiredFile in @('metrics.json', 'evaluation_predictions.csv')) {
    if (-not (Test-Path -LiteralPath (Join-Path $artifactDirectory $requiredFile) -PathType Leaf)) {
        throw "Azure output is missing $requiredFile beside model.joblib. Downloaded files are in $downloadPath"
    }
}

foreach ($fileName in @('model.joblib', 'metrics.json', 'evaluation_predictions.csv')) {
    Copy-Item -LiteralPath (Join-Path $artifactDirectory $fileName) `
        -Destination (Join-Path $outputsPath $fileName) -Force
}
$sourceInfo = @{
    source = 'Azure Machine Learning'
    job_name = $JobName
    imported_at = (Get-Date).ToUniversalTime().ToString('o')
} | ConvertTo-Json
Set-Content -LiteralPath (Join-Path $outputsPath 'active_model.json') -Value $sourceInfo -Encoding utf8

Write-Host 'Azure-trained model and evaluation files are now active in the local app.'
Write-Host 'Refresh the Streamlit page to load the imported model.'
Write-Host "Downloaded artifacts retained at: $downloadPath"
