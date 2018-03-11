#!/bin/bash
set -e

which cfget >/dev/null || { echo "cfget not installed; try 'sudo apt install cfget'"; exit 1; }
which socat >/dev/null || { echo "socat not installed; try 'sudo apt install socat'"; exit 1; }

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPTDIR}"

ARDUINO_DEV=$(cfget -C ino.ini serial-port)

./configure_port_speed.sh
#socat -d -d -,raw,echo=0 $ARDUINO_DEV,raw,echo=0
socat -d -d -,echo=0 ${ARDUINO_DEV},raw,echo=0
