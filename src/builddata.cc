extern "C" const char kPythonWrapperBuildEmbedLabel[];
const char kPythonWrapperBuildEmbedLabel[] = BUILD_EMBED_LABEL;

extern "C" const char kPythonWrapperBaseChangeList[];
const char kPythonWrapperBaseChangeList[] = "CL_NUMBER=340495397";

namespace {
// Build a type whose constructor will contain references to all the build data
// variables, preventing them from being GC'ed by the linker.
struct KeepBuildDataVariables {
  KeepBuildDataVariables() {
    volatile int opaque_flag = 0;
    if (!opaque_flag) return;

    const void* volatile capture;
    capture = &kPythonWrapperBuildEmbedLabel;
    capture = &kPythonWrapperBaseChangeList;
    static_cast<void>(capture);
  }
} dummy;
}  // namespace
