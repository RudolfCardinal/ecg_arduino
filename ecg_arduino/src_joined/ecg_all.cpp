// This file includes all the .cpp files required for the project, so that
// it works with abuild.bat under Windows.


// Under Windows, TimerOne.h needs to use Arduino.h, not WProgram.h, so the
// next line is a hack to do that. We can't use an #ifdef to detect Windows,
// since avr-g++ is configured to compile for AVR.
#define ARDUINO 100

#include "../src/TimerOne.cpp"
#include "../src/ecg.cpp"
