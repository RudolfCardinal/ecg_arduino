#!/bin/bash

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPTDIR}"

ARDUINO_DEV=$(cfget -C ino.ini serial-port)

./configure_port_speed.sh
#socat -d -d -,raw,echo=0 $ARDUINO_DEV,raw,echo=0
socat -d -d -,echo=0 ${ARDUINO_DEV},raw,echo=0
