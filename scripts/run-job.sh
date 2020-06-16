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

# trapCleanup Function
# -----------------------------------
# Any actions that should be taken if the script is prematurely
# exited.  Always call this function at the top of your script.
# -----------------------------------
function trapCleanup() {
  echo ""
  # If tmpDir exists, remove it
  if [ -d "${tmpDir}" ]; then
    rm -r "${tmpDir}"
  fi
  exit 1;
}

# Set Flags
# -----------------------------------
# Flags which can be overridden by user input.
# Default values are below
# -----------------------------------
quiet=0
printLog=0
verbose=0
force=0
strict=0
debug=0
args=()

# Set Temp Directory
# -----------------------------------
# Create temp directory with three random numbers and the process ID
# in the name.  This directory is removed automatically at exit.
# -----------------------------------
tmpDir="/tmp/${scriptName}.$RANDOM.$RANDOM.$RANDOM.$$"
(umask 077 && mkdir "${tmpDir}") || {
  die "Could not create temporary directory! Exiting."
}

# Logging
# -----------------------------------
# Log is only used when the '-l' flag is set.
#
# To never save a logfile change variable to '/dev/null'
# Save to Desktop use: $HOME/Desktop/${scriptBasename}.log
# Save to standard user log location use: $HOME/Library/Logs/${scriptBasename}.log
# -----------------------------------
logFile="$HOME/Library/Logs/${scriptBasename}.log"


function mainScript() {
############## Begin Script Here ###################
####################################################

# Load variables from the config file
if test -f "$file"; then
  eval "$(parse_yaml "$file")"
else
  eval "$(parse_yaml default-config.yml)"
fi

options=""
if [ -n "$width" ]; then
  options+=" --width "
  options+=$width
fi
if [ -n "$height" ]; then
  options+=" --height "
  options+=$height
fi
if [ -n "$ball" ]; then
  options+=" --ball "
  options+=$ball
fi
if [ -n "$snell" ]; then
  options+=" --snell "
  options+=$snell
fi
if [ -n "$paddle_speed" ]; then
  options+=" --paddle-speed "
  options+=$paddle_speed
fi
if [ -n "$paddle_length" ]; then
  options+=" --paddle-length "
  options+=$paddle_length
fi
if [ -n "$learning_rate" ]; then
  options+=" --learning-rate "
  options+=$learning_rate
fi
if [ -n "$update_prob" ]; then
  options+=" --update-prob "
  options+=$update_prob
fi
if [ -n "$episodes" ]; then
  options+=" --episodes "
  options+=$episodes
fi
if [ -n "$resume" ]; then
  options+=" --resume "
  options+=$resume
fi
if [ -n "$checkpoint" ]; then
  options+=" --checkpoint "
  options+=$checkpoint
fi
if [ -n "$history" ]; then
  options+=" --history "
  options+=$history
fi

options+=" --store-dir ../experiments/"
options+="$name"

if [ -z "$user" ]; then
  die "--user must be supplied"
fi
if [ -z "$name" ]; then
  die "--name must be supplied"
fi

# If branch not specified, set to master
if [ -z "$branch" ]; then
  branch="master"
fi

printf -v job_name "%s-%s" "$user" "$name"
export job_name
export options
export branch

source /dev/stdin <<<"$(echo 'cat <<EOF >final.yml'; cat job.yml; echo EOF;)"

cat final.yml | kubectl create -f -

rm -f final.yml

####################################################
############### End Script Here ####################
}

############## Begin Options and Usage ###################


# Print usage
usage() {
  echo -n "${scriptName} [OPTION]...

Run kubernetes job

 Options:
  -u, --user        Must correspond to /data/<user>/
  -n, --name        Job name to append
  -b, --branch      Branch to run (default: master)
  -f, --file        Config file (default: experiment-config.yml)
  -q, --quiet       Quiet (no output)
  -l, --log         Print log to file
  -v, --verbose     Output more information. (Items echoed to 'verbose')
  -d, --debug       Runs script in BASH debug mode (set -x)
  -h, --help        Display this help and exit
      --version     Output version information and exit
"
}

# Iterate over options breaking -ab into -a -b when needed and --foo=bar into
# --foo bar
optstring=h
unset options
while (($#)); do
  case $1 in
    # If option is of type -ab
    -[!-]?*)
      # Loop over each character starting with the second
      for ((i=1; i < ${#1}; i++)); do
        c=${1:i:1}

        # Add current char to options
        options+=("-$c")

        # If option takes a required argument, and it's not the last char make
        # the rest of the string its argument
        if [[ $optstring = *"$c:"* && ${1:i+1} ]]; then
          options+=("${1:i+1}")
          break
        fi
      done
      ;;

    # If option is of type --foo=bar
    --?*=*) options+=("${1%%=*}" "${1#*=}") ;;
    # add --endopts for --
    --) options+=(--endopts) ;;
    # Otherwise, nothing special
    *) options+=("$1") ;;
  esac
  shift
done
set -- "${options[@]}"
unset options

# Print help if no arguments were passed.
# Uncomment to force arguments when invoking the script
# [[ $# -eq 0 ]] && set -- "--help"

# Read the options and set stuff
while [[ $1 = -?* ]]; do
  case $1 in
    -h|--help) usage >&2; exit 0 ;;
    --version) echo "$(basename $0) ${version}"; exit 0 ;;
    -u|--user) shift; user=${1} ;;
    -n|--name) shift; name=${1} ;;
    -b|--branch) shift; branch=${1} ;;
    -f|--file) shift; file=${1} ;;
    -v|--verbose) verbose=1 ;;
    -l|--log) printLog=1 ;;
    -q|--quiet) quiet=1 ;;
    -s|--strict) strict=1;;
    -d|--debug) debug=1;;
    --force) force=1 ;;
    --endopts) shift; break ;;
    *) die "invalid option: '$1'." ;;
  esac
  shift
done

# Store the remaining part as arguments.
args+=("$@")

############## End Options and Usage ###################




# ############# ############# #############
# ##       TIME TO RUN THE SCRIPT        ##
# ##                                     ##
# ## You shouldn't need to edit anything ##
# ## beneath this line                   ##
# ##                                     ##
# ############# ############# #############

# Trap bad exits with your cleanup function
trap trapCleanup EXIT INT TERM

# Exit on error. Append '||true' when you run the script if you expect an error.
set -o errexit

# Run in debug mode, if set
if [ "${debug}" == "1" ]; then
  set -x
fi

# Exit on empty variable
if [ "${strict}" == "1" ]; then
  set -o nounset
fi

# Bash will remember & return the highest exitcode in a chain of pipes.
# This way you can catch the error in case mysqldump fails in `mysqldump |gzip`, for example.
set -o pipefail

# Run your script
mainScript

exit 0 # Exit cleanly