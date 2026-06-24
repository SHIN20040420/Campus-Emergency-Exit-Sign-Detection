param(
    [int]$PositiveCount = 80,
    [int]$NegativeCount = 120,
    [int]$Width = 48,
    [int]$Height = 24,
    [int]$Stages = 12
)

$ErrorActionPreference = "Stop"

$TrainingDir = "training"
$CascadeDir = Join-Path $TrainingDir "cascade"
$PositiveInfo = Join-Path $TrainingDir "positives.txt"
$NegativeInfo = Join-Path $TrainingDir "negatives.txt"
$VectorFile = Join-Path $TrainingDir "positives.vec"

if (!(Test-Path $PositiveInfo)) {
    throw "Missing $PositiveInfo. Run: python src/prepare_haar_data.py --dataset dataset --output training"
}

if (!(Test-Path $NegativeInfo)) {
    throw "Missing $NegativeInfo. Run: python src/prepare_haar_data.py --dataset dataset --output training"
}

New-Item -ItemType Directory -Force -Path $CascadeDir | Out-Null

opencv_createsamples `
    -info $PositiveInfo `
    -vec $VectorFile `
    -num $PositiveCount `
    -w $Width `
    -h $Height

opencv_traincascade `
    -data $CascadeDir `
    -vec $VectorFile `
    -bg $NegativeInfo `
    -numPos $PositiveCount `
    -numNeg $NegativeCount `
    -numStages $Stages `
    -w $Width `
    -h $Height `
    -featureType HAAR `
    -mode ALL

Write-Host "Done. Cascade file: $CascadeDir/cascade.xml"
