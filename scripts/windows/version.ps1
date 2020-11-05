param (
    $File
)

function Get-ResourceString {
    param (
        $Key,
        $File
    )
    $FileContent = ((Get-Content $File) -replace "`0", "")
    $Lines = foreach ($Line in $FileContent) {
        if ($Line -match "($Key)") {
            $Line
        }
    }
    $Lines = foreach ($Line in $Lines) {
        if ($Line -match "($Key).*$") {
            $Matches[0]
        }
    }
    $Lines = $Lines -replace '[\W]', "`r`n"
    $Line = foreach ($Line in $Lines) {
        if ($Line -match "($Key).*\s") {
            $Matches[0]
        }
    }
    $Rest = $Line.Replace($Key, "")
    $Output = "$Key`: $Rest".Trim()
    Write-Output $Output
}

Get-ResourceString -File $File -Key 'CL_NUMBER'
Get-ResourceString -File $File -Key 'TENSORFLOW_COMMIT'
