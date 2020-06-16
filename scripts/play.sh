#!/usr/bin/env bash

# ##################################################
# My Generic BASH script template
#
version="1.0.0"
#
# ##################################################

# path to this script's parent directory.
scriptPath="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Source Scripting Utilities
# -----------------------------------
# These shared utilities provide many functions which are needed to provide
# the functionality in this boilerplate. This script will fail if they can
# not be found.
# -----------------------------------

utilsLocation="${scriptPath}/utils.sh" # Update this path to find the utilities.

if [ -f "${utilsLocation}" ]; then
  source "${utilsLocation}"
else
  echo "Find the file utils.sh and add a reference to it in this script. Exiting."
  exit 1
fi

# read yaml file
eval $(parse_yaml experiment-config.yml)

# access yaml content
echo $resume
