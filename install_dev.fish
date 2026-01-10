#!/usr/bin/env fish
# install_dev.fish
# Run from anywhere; installs this repo's requirements + editable install into the current environment.

function _die
    echo "install_dev.fish: $argv" 1>&2
    exit 1
end

# Locate the directory this script lives in (repo root if you place it there)
set -l script_path (status --current-filename)
test -n "$script_path"; or _die "Can't determine script path (status --current-filename is empty)."

set -l repo_dir (realpath (dirname "$script_path"))
test -d "$repo_dir"; or _die "Repo dir not found: $repo_dir"

set -l req_file "$repo_dir/requirements.txt"
test -f "$req_file"; or _die "Missing requirements.txt at: $req_file"

command -qs uv; or _die "uv not found on PATH. Install uv first (e.g. pipx install uv)."

echo "Repo: $repo_dir"
echo "Using requirements: $req_file"
echo

# Install deps + editable install of THIS repo (portable, no absolute hardcoding)
uv pip install -r "$req_file"; or _die "uv pip install -r failed"
uv pip install -e "$repo_dir"; or _die "uv pip install -e failed"

echo
echo "Done: installed requirements + editable package from $repo_dir"
