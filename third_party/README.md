# Third-party dependencies

This directory hosts external dependencies vendored via git submodules.

## whisper.cpp

Path: `third_party/whisper.cpp`
Repository: https://github.com/ggml-org/whisper.cpp

Clone the submodule after checking out this repository:

```bash
git submodule update --init --recursive
```

To upgrade the dependency:

```bash
git fetch --recurse-submodules
# optionally checkout a new tag or commit inside third_party/whisper.cpp
# then stage and commit the updated submodule reference
```

The native bindings rely on the optional `whispercpp` build tag. Local development without
initialising the submodule still works (stub engine is used), but release builds must ensure
`third_party/whisper.cpp` is present and compiled.
