#!/usr/bin/env bash

# Maybe change the UID/GID of the container's non-root user to match the host's UID/GID

: "${REMOTE_USER:="coder"}";
: "${OLD_UID:=}";
: "${OLD_GID:=}";
: "${NEW_UID:=}";
: "${NEW_GID:=}";

OLD_UID="$(sed -n "s/${REMOTE_USER}:[^:]*:\([^:]*\):\([^:]*\):[^:]*:\([^:]*\).*/\1/p" /etc/passwd)";
OLD_GID="$(sed -n "s/${REMOTE_USER}:[^:]*:\([^:]*\):\([^:]*\):[^:]*:\([^:]*\).*/\2/p" /etc/passwd)";
HOME_FOLDER="$(sed -n "s/${REMOTE_USER}:[^:]*:\([^:]*\):\([^:]*\):[^:]*:\([^:]*\).*/\3/p" /etc/passwd)";
EXISTING_USER="$(sed -n "s/\([^:]*\):[^:]*:${NEW_UID}:.*/\1/p" /etc/passwd)";
EXISTING_GROUP="$(sed -n "s/\([^:]*\):[^:]*:${NEW_GID}:.*/\1/p" /etc/group)";

if [[ -z "${OLD_UID}" ]]; then
    echo "Remote user not found in /etc/passwd (${REMOTE_USER}).";
    entry_script="$(pwd)/.devcontainer/cccl-entrypoint.sh";
    exec "${entry_script}" "$@";
elif [[ "${OLD_UID}" = "${NEW_UID}" ]] && [[ "${OLD_GID}" = "${NEW_GID}" ]]; then
    echo "UIDs and GIDs are the same (${NEW_UID}:${NEW_GID}).";
    # Even when IDs match, ensure we execute as the non-root REMOTE_USER so
    # gh and sccache use the mapped HOME (/home/coder) where ~/.aws is bind-mounted.
    export VIRTUAL_ENV=;
    export VIRTUAL_ENV_PROMPT=;
    export HOME="${HOME_FOLDER}";
    export XDG_CACHE_HOME="${HOME_FOLDER}/.cache";
    export XDG_CONFIG_HOME="${HOME_FOLDER}/.config";
    export XDG_STATE_HOME="${HOME_FOLDER}/.local/state";
    export PYTHONHISTFILE="${HOME_FOLDER}/.local/state/.python_history";
    entry_script="$(pwd)/.devcontainer/cccl-entrypoint.sh"
    exec su -p "${REMOTE_USER}" -- "${entry_script}" "$@";
elif [[ "${OLD_UID}" != "${NEW_UID}" ]] && [[ -n "${EXISTING_USER}" ]]; then
    echo "User with UID exists (${EXISTING_USER}=${NEW_UID}).";
    entry_script="$(pwd)/.devcontainer/cccl-entrypoint.sh";
    exec "${entry_script}" "$@";
else
    if [[ "${OLD_GID}" != "${NEW_GID}" ]] && [[ -n "${EXISTING_GROUP}" ]]; then
        echo "Group with GID exists (${EXISTING_GROUP}=${NEW_GID}).";
        NEW_GID="${OLD_GID}";
    fi
    echo "Updating UID:GID from ${OLD_UID}:${OLD_GID} to ${NEW_UID}:${NEW_GID}.";
    sed -i -e "s/\(${REMOTE_USER}:[^:]*:\)[^:]*:[^:]*/\1${NEW_UID}:${NEW_GID}/" /etc/passwd;
    if [[ "${OLD_GID}" != "${NEW_GID}" ]]; then
        sed -i -e "s/\([^:]*:[^:]*:\)${OLD_GID}:/\1${NEW_GID}:/" /etc/group;
    fi

    num_proc="$(nproc --all)"
    # Fast parallel `chown -R`
    find "${HOME_FOLDER}/" -not -user "${REMOTE_USER}" -print0 \
  | xargs -0 -r -n1 -P"${num_proc}" chown "${NEW_UID}:${NEW_GID}"

    # Run the container command as $REMOTE_USER, preserving the container startup environment.
    #
    # We cannot use `su -w` because that's not supported by the `su` in Ubuntu18.04, so we reset the following
    # environment variables to the expected values, then pass through everything else from the startup environment.
    export VIRTUAL_ENV=;
    export VIRTUAL_ENV_PROMPT=;
    export HOME="${HOME_FOLDER}";
    export XDG_CACHE_HOME="${HOME_FOLDER}/.cache";
    export XDG_CONFIG_HOME="${HOME_FOLDER}/.config";
    export XDG_STATE_HOME="${HOME_FOLDER}/.local/state";
    export PYTHONHISTFILE="${HOME_FOLDER}/.local/state/.python_history";

    if command -V module 2>&1 | grep -q function; then
        # "deactivate" lmod so it will be reactivated as the non-root user
        export LMOD_CMD=
        export LMOD_DEFAULT_MODULEPATH=
        export LMOD_DIR=
        export LMOD_PKG=
        export LOADEDMODULES=
        export MANPATH=
        export MODULEPATH_ROOT=
        export MODULEPATH=
        export MODULESHOME=
        export -fn module
    fi

    entry_script="$(pwd)/.devcontainer/cccl-entrypoint.sh"
    exec su -p "${REMOTE_USER}" -- "${entry_script}" "$@";
fi
