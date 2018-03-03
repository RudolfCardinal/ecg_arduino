#!/bin/bash

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPTDIR}"

ARDUINO_DEV=$(cfget -C ino.ini serial-port)
ARDUINO_BAUD=$(cfget -C ino.ini baud-rate)

# Configure for 1 Mbit/s (or whatever), 8N1
stty -F ${ARDUINO_DEV} ${ARDUINO_BAUD} cs8 -ignpar -cstopb

# Report current configuration
stty -F ${ARDUINO_DEV}

