param (
    [Parameter(Mandatory=$true)]
    [string]$ImagePath,
    [string]$OutputPath = ""
)

$ImagePath = [System.IO.Path]::GetFullPath($ImagePath)
if (!(Test-Path $ImagePath)) {
    Write-Error "Image file not found: $ImagePath"
    exit 1
}

Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Runtime.WindowsRuntime

$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Foundation.IAsyncOperation`1, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.InMemoryRandomAccessStream, Windows.Storage.Streams, ContentType = WindowsRuntime]

$getAwaiterBaseMethod = [WindowsRuntimeSystemExtensions].GetMember('GetAwaiter').Where({ 
    $PSItem.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' 
}, 'First')[0]

function Invoke-Async {
    param($AsyncTask, $ResultType)
    $genericMethod = $getAwaiterBaseMethod.MakeGenericMethod($ResultType)
    $awaiter = $genericMethod.Invoke($null, @($AsyncTask))
    return $awaiter.GetResult()
}

$bytes = [System.IO.File]::ReadAllBytes($ImagePath)
$inMemStream = [Windows.Storage.Streams.InMemoryRandomAccessStream]::new()
$writer = [Windows.Storage.Streams.DataWriter]::new($inMemStream)
$writer.WriteBytes($bytes)
$null = $writer.StoreAsync().GetResults()
$writer.DetachStream() | Out-Null
$inMemStream.Seek(0)

$decoderTask = [Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($inMemStream)
$decoder = Invoke-Async $decoderTask ([Windows.Graphics.Imaging.BitmapDecoder])

$softwareBitmapTask = $decoder.GetSoftwareBitmapAsync()
$softwareBitmap = Invoke-Async $softwareBitmapTask ([Windows.Graphics.Imaging.SoftwareBitmap])

$ocrEngine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if ($null -eq $ocrEngine) {
    $lang = New-Object Windows.Globalization.Language("en")
    $ocrEngine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($lang)
}

if ($null -eq $ocrEngine) {
    Write-Error "Windows OCR Engine could not be initialized."
    exit 1
}

$ocrResultTask = $ocrEngine.RecognizeAsync($softwareBitmap)
$ocrResult = Invoke-Async $ocrResultTask ([Windows.Media.Ocr.OcrResult])

$wordsList = @()
foreach ($line in $ocrResult.Lines) {
    foreach ($word in $line.Words) {
        $wordsList += [PSCustomObject]@{
            Text = $word.Text
            X = [int]$word.BoundingRect.X
            Y = [int]$word.BoundingRect.Y
            Width = [int]$word.BoundingRect.Width
            Height = [int]$word.BoundingRect.Height
        }
    }
}

$json = $wordsList | ConvertTo-Json -Compress
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

if ($OutputPath -ne "") {
    $wordsList | ConvertTo-Json -Depth 4 | Set-Content $OutputPath -Encoding UTF8
}

[Console]::WriteLine($json)
