#!/bin/bash

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPTDIR}"

# Kill socat
# kill $(ps aux | grep '[s]ocat' | awk '{print $2}')
# http://stackoverflow.com/questions/3510673/find-and-kill-a-process-in-one-line-using-bash-and-regex

./build_upload.sh && ./term.sh

