#!/bin/bash

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPTDIR}"

ARDUINO_DEV=$(cfget -C ino.ini serial-port)

ino build && ino upload -p ${ARDUINO_DEV}
