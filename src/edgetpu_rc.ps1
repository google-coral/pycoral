param (
  $BuildStatusFile,
  $BuildDataFile,
  $ResFileTemplate,
  $OutputFile
)

$BuildStatus = Get-Content $BuildStatusFile
$BuildData = Get-Content $BuildDataFile
$ResFile = Get-Content $ResFileTemplate
$ClNumber = (((-split ($BuildData -match 'CL_NUMBER'))[-1]) -split '=')[-1].Trim("`";")
$TensorflowCommit = (-split ($BuildStatus -match 'BUILD_EMBED_LABEL'))[-1]
$ResFile = $ResFile.Replace('CL_NUMBER_TEMPLATE', $ClNumber)
$ResFile = $ResFile.Replace('TENSORFLOW_COMMIT_TEMPLATE', $TensorflowCommit)
Out-File -FilePath $OutputFile -InputObject $ResFile -Encoding unicode
