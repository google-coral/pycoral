// Redefine some constants we would usually get from winver.h
// Bazel doesn't know the correct include paths to pass along
// to rc to pick up the header.
#define VS_VERSION_INFO      1
#define VS_FFI_FILEFLAGSMASK 0x3FL
#define VS_FF_DEBUG          0x1L
#define VOS__WINDOWS32       0x4L
#define VFT_DLL              0x2L

#define CL_NUMBER_STR "CL_NUMBER_TEMPLATE\040"
#define TENSORFLOW_COMMIT_STR "TENSORFLOW_COMMIT_TEMPLATE\040"

VS_VERSION_INFO VERSIONINFO
FILEFLAGSMASK VS_FFI_FILEFLAGSMASK
FILEFLAGS 0
FILEOS VOS__WINDOWS32
FILETYPE VFT_DLL
FILESUBTYPE 0
BEGIN
  BLOCK "StringFileInfo"
  BEGIN
    BLOCK "040904E4"
    BEGIN
      VALUE "FileDescription", "EdgeTPU Python library\0"
      VALUE "InternalName", "_pywrap_coral.pyd\0"
      VALUE "LegalCopyright", "(C) 2019-2020 Google, LLC\0"
      VALUE "ProductName", "edgetpu\0"
      VALUE "CL_NUMBER", CL_NUMBER_STR
      VALUE "TENSORFLOW_COMMIT", TENSORFLOW_COMMIT_STR
    END
  END
  BLOCK "VarFileInfo"
  BEGIN
    VALUE "Translation", 0x0409, 1252
  END
END
