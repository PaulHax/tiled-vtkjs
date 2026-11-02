#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source ./vtkjs-fork.env

if [[ ! "$VTKJS_FORK_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  printf '%s\n' 'VTKJS_FORK_COMMIT must be a full commit SHA' >&2
  exit 1
fi

build_directory="$(mktemp -d "${TMPDIR:-/tmp}/tiled-vtkjs-netlify.XXXXXX")"
vtk_source="$build_directory/vtk-js"
vtk_esm="$vtk_source/dist/esm"
vtk_link="$project_root/node_modules/@kitware/vtk.js"
vtk_backup="$build_directory/previous-vtk.js"
previous_dependency=false
temporary_link=false

cleanup() {
  local exit_status=$?
  trap - EXIT
  if [[ "$temporary_link" == true ]]; then
    if [[ -L "$vtk_link" && "$(readlink "$vtk_link")" == "$vtk_esm" ]]; then
      rm "$vtk_link"
    else
      printf 'VTK dependency changed during the build; preserving %s\n' "$build_directory" >&2
      exit 1
    fi
  fi
  if [[ "$previous_dependency" == true ]]; then
    mv "$vtk_backup" "$vtk_link"
  fi
  rm -rf "$build_directory"
  exit "$exit_status"
}
trap cleanup EXIT

# Netlify's automatic dependency install skips lifecycle scripts until VTK is ready.
npm rebuild
git init "$vtk_source"
git -C "$vtk_source" remote add origin "$VTKJS_FORK_REPO"
git -C "$vtk_source" fetch --depth 1 origin "$VTKJS_FORK_COMMIT"
git -C "$vtk_source" checkout --detach FETCH_HEAD
if [[ "$(git -C "$vtk_source" rev-parse HEAD)" != "$VTKJS_FORK_COMMIT" ]]; then
  printf '%s\n' 'Fetched vtk.js revision does not match VTKJS_FORK_COMMIT' >&2
  exit 1
fi
npm --prefix "$vtk_source" ci
npm --prefix "$vtk_source" run build:esm
test -s "$vtk_esm/Rendering/Core/PointGaussianMapper.d.ts"
test -s "$vtk_esm/Rendering/Core/Texture.d.ts"

mkdir -p "$(dirname "$vtk_link")"
# Preserve an existing local dependency, including a dangling cached symlink.
if [[ -e "$vtk_link" || -L "$vtk_link" ]]; then
  mv "$vtk_link" "$vtk_backup"
  previous_dependency=true
fi
ln -s "$vtk_esm" "$vtk_link"
temporary_link=true
npm run prepare
VTK_JS_DIR="$vtk_esm" npm run example:build
