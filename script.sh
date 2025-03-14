# generate_crops.ps1
Set-Location -Path "C:\Users\jobin\agrobot"
New-Item -ItemType Directory -Path "data/crops" -Force | Out-Null

for ($i = 1; $i -le 200; $i++) {
    $cropName = "Crop$i"
    $tempMin = Get-Random -Minimum -5 -Maximum 15
    $tempMax = $tempMin + (Get-Random -Minimum 10 -Maximum 25)
    $humidMin = Get-Random -Minimum 30 -Maximum 60
    $humidMax = $humidMin + (Get-Random -Minimum 20 -Maximum 40)
    $fertilizers = @("Urea", "Diammonium Phosphate", "Potassium Chloride", "Ammonium Sulfate") | Get-Random -Count (Get-Random -Minimum 2 -Maximum 4)

    $cropData = @"
    {
        "name": "$cropName",
        "weather_preferences": {
            "temperature": {"min": $tempMin, "max": $tempMax},
            "humidity": {"min": $humidMin, "max": $humidMax}
        },
        "fertilizers": $(ConvertTo-Json $fertilizers -Compress)
    }
    "@
    Set-Content -Path "data/crops/$cropName.json" -Value $cropData
}

Write-Host "Generated 200 crop entries in data/crops/"