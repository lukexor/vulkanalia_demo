#!/usr/bin/env bash

echo "compiling shaders..."

if ! command -v glslc &> /dev/null; then
  echo "glslc is not installed or cannot be located. the Vulkan SDK may not be installed correctly."
fi

pushd "$(dirname "$0")" > /dev/null || exit
/usr/bin/env glslc shader.vert -o vert.spv
/usr/bin/env glslc shader.frag -o frag.spv
popd > /dev/null || exit

echo "done!"
