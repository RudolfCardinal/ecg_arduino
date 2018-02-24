/*
    Arduino ECG controller.
    Copyright (C) 2018-2018 Rudolf Cardinal (rudolf@pobox.com).

    This is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This software is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this software. If not, see <http://www.gnu.org/licenses/>.
*/

// #define DEBUG

#include <Arduino.h>

const char strVersion[] = "Arduino ECG controller v0.1, by Rudolf Cardinal, 2018-02-24";
const int ECG_PIN = A1;
const int VOLTAGE_RANGE = DEFAULT;  // 3.3 V reference
const char strHardwareInfo[] = "Expects a device providing 0 to +3.3V on Arduino analogue input pin A1";

/*
===============================================================================
VERSION HISTORY
===============================================================================

- v0.1, completed 2018-02-19.

===============================================================================
PROTOCOL
===============================================================================

-------------------------------------------------------------------------------
1. Principles
-------------------------------------------------------------------------------

- Use a high-speed serial interface over USB. Don't use 9600 bps! Use 1 Mbit/s.
- All messages from the Arduino to the computer are line-terminated.
- All messages from the computer to the Arduino are line-terminated.
- Keep a minimum of legibility: spaces between commands/parameters.
- Allow leading/trailing spaces in input.
- Otherwise, keep it minimalistic for serial speed.
- Case-sensitive.

-------------------------------------------------------------------------------
2. Commands from computer to Arduino
-------------------------------------------------------------------------------

E

    Enter ECG mode.

S   <freq_hz>

    Enter sine wave mode.

-------------------------------------------------------------------------------
3. Responses and spontaneous events from Arduino to computer
-------------------------------------------------------------------------------

<time>,<value>

    Times: in microseconds.
        Why send them? Because the Arduino interrupt timers are NOT
        SUFFICIENTLY ACCURATE to avoid timestamps -- sometimes grossly
        inaccurate, it seems, or FlexiTimer2.cpp is dodgy, or I'm using it
        wrong. -- Checked: it's dodgy.
    Analogue values: integers in the range 0-1023 inclusive.

===============================================================================
MORE ON TIMERS, given that FlexiTimer2 is buggy.
===============================================================================

Hardware:
    https://www.robotshop.com/letsmakerobots/arduino-101-timers-and-interrupts

A good library instead:
    https://www.pjrc.com/teensy/td_libs_TimerOne.html

*/

#include <avr/boot.h>
#include <avr/pgmspace.h>
#include <util/atomic.h> // for ATOMIC_BLOCK
// #include "FlexiTimer2.h"  // https://github.com/wimleers/flexitimer2
#include "TimerOne.h"

#define BAUD_RATE 1000000  // 1 Mbit/s
#define SERIAL_DATA_PARITY_STOP SERIAL_8N1

#define INPUT_BUFSIZE 20  // DO NOT USE 100.
/*
MEMORY IS TIGHT. SEE whiskard.cpp
REMEMBER: IF BEHAVIOUR IS ODD, CHECK THE MEMORY USAGE.
*/

#define BASE_10 10

const char strError[] = "Error: ";
const char strWarning[] = "Warning: ";
const char strInfo[] = "Info: ";
const char strDebug[] = "Debug: ";
const char strIntegerExpected[] = "Integer expected";
const char strFloatExpected[] = "Float expected";
const char strRedundantParameters[] = "Redundant parameters";
const char strCommandReceived[] = "Command received: ";
const char strBlankCommand[] = "Blank command";
const char strInputBufferOverflow[] = "Input buffer overflow";
const char strSampleOverflow[] = "Sample overflow";
const char strUnknownCommand[] = "Unknown command";
const char respOK[] = "OK";
const char respHello[] = "Hello";

const char cmdSourceAnalogue[] = "A";           // A
const char cmdSourceSineGenerator[] = "G";      // G
const char cmdSourceTime[] = "T";               // T
const char cmdSineGeneratorFrequency[] = "S";   // S <float frequency_hz>
const char cmdSampleFrequency[] = "F";          // F <float frequency_hz>
const char cmdSampleContinuous[] = "C";         // C
const char cmdSampleNumber[] = "N";             // N <int number_of_samples>
const char cmdQuiet[] = "Q";                    // Q

char incoming[INPUT_BUFSIZE];  // incoming itself is a const pointer to a non-const array of char.
char nextword[INPUT_BUFSIZE];  // http://stackoverflow.com/questions/3839553/array-as-const-pointer
unsigned char incomingIndex = 0;

const unsigned char SOURCE_ECG = 0;
const unsigned char SOURCE_SINE = 1;
const unsigned char SOURCE_TIME = 2;

const unsigned char NUM_DP_HZ = 4;
const unsigned char NUM_DP_S_FOR_MICROSEC = 6;

unsigned char debug_level;
volatile unsigned char source;
volatile float sine_freq_hz;            // >8 bits; needs ATOMIC_BLOCK (*)
    // (*) see https://www.arduino.cc/reference/en/language/variables/variable-scope--qualifiers/volatile/
volatile float sine_time_s;             // >8 bits; needs ATOMIC_BLOCK
volatile float sine_phase_duration_s;   // >8 bits; needs ATOMIC_BLOCK
volatile bool sample_pending;
volatile unsigned long sample_time;     // >8 bits; needs ATOMIC_BLOCK
volatile unsigned long sample_value;    // >8 bits; needs ATOMIC_BLOCK
volatile bool continuous_sampling;
volatile unsigned int samples_to_go;    // >8 bits; needs ATOMIC_BLOCK
volatile bool missed_sample;

const float MAX_OUTPUT = 1024;
const float HALF_OUTPUT = MAX_OUTPUT / 2;
const float MAX_TIMER_RESOLUTION_S = 2;
    // for FlexiTimer2, we needed to use 2 ms (or not much higher)

bool timer1_initialized;
float sampling_freq_hz;
double sampling_resolution_s;

volatile unsigned long timer_units;
volatile unsigned long timer_count;
volatile bool timer_overflowing;

const float DEFAULT_SINE_FREQUENCY_HZ = 0.1;
const float DEFAULT_SAMPLING_FREQUENCY_HZ = 100;

// PI, HALF_PI, and TWO_PI are predefined in Arduino.h


//=============================================================================
// Comms to computer
//=============================================================================

void error(const char* str)
{
    Serial.print(strError);
    Serial.println(str);
}

void warning(const char* str)
{
    Serial.print(strWarning);
    Serial.println(str);
}

void info(const char* str)
{
    Serial.print(strInfo);
    Serial.println(str);
}

void debug(const char* str)
{
    Serial.print(strDebug);
    Serial.println(str);
}

void sendValue(unsigned long time, int value)
{
    // time,value
    Serial.print(time);
    Serial.print(",");
    Serial.println(value);
}

void sendValue(unsigned long time, unsigned long value)
{
    // time,value
    Serial.print(time);
    Serial.print(",");
    Serial.println(value);
}


//=============================================================================
// Parsing ancillary functions
//=============================================================================

void resetInputBuffers()
{
    incomingIndex = 0;
    *incoming = 0;  // null-terminate the empty buffer
    *nextword = 0;  // and this one
}

bool getIntFromWord(const char* nptr, int& value)
{
    char* end;
    if (!*nptr) {
        // empty
        return false;
    }
    long int longnumber = strtol(nptr, &end, BASE_10);
    if (*end) {
        // something invalid
        return false;
    }
    value = (int) longnumber;
    return true;
}

bool getFloatFromWord(const char* nptr, float& value)
{
    char* end;
    if (!*nptr) {
        // empty
        return false;
    }
    double result = strtod(nptr, &end);
    if (*end) {
        // something invalid
        return false;
    }
    value = (float) result;
    return true;
}

void getNextWord(char*& from, char* to)
{
    // Skip over any spaces in the input
    while (*from == ' ') {
        ++from;
    }
    // Copy a word, up to a space or the end of the input string.
    // Relies on the assurance that the destination buffer is big enough.
    // (That follows from the definition of "incoming" and "nextword" as
    // having the same size.)
    while (*from && *from != ' ') {
        *to = *from;
        ++to;
        ++from;
    }
    // Null-terminate the output
    *to = 0;
}

bool getNextInt(char*& from, int& value)
{
    getNextWord(from, nextword);
    if (!getIntFromWord(nextword, value)) {
        error(strIntegerExpected);
        return false;
    }
    return true;
}

bool getNextFloat(char*& from, float& value)
{
    getNextWord(from, nextword);
    if (!getFloatFromWord(nextword, value)) {
        error(strFloatExpected);
        return false;
    }
    return true;
}

bool redundantParameters(char*& from)
{
    getNextWord(from, nextword);
    if (*nextword) {
        error(strRedundantParameters);
        return true;
    }
    return false;
}


//=============================================================================
// Helper functions
//=============================================================================

void setSineFrequency(float frequency_hz)
{
    if (debug_level >= 2) {
        Serial.print(strDebug);
        Serial.print("Setting sine wave generator: frequency (Hz) = ");
        Serial.println(frequency_hz, NUM_DP_HZ);
    }
    ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
        sine_freq_hz = frequency_hz;
        sine_phase_duration_s = 1.0 / sine_freq_hz;
        sine_time_s = 0;
    }
}


void setSampling(bool continuous, int num_samples)
{
    if (debug_level >= 2) {
        Serial.print(strDebug);
        Serial.print("Setting sampling: continuous = ");
        Serial.print(continuous);
        Serial.print(", num_samples = ");
        Serial.print(num_samples);
        Serial.print(", start time = ");
        Serial.println(micros());
    }
    ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
        continuous_sampling = continuous;
        samples_to_go = num_samples;
        missed_sample = false;
        sample_pending = false;
    }
}

void timerISRMain()
{
    // - We're in an interrupt here.
    // - Don't use the serial port. Its I2C protocol uses interrupts.
    // - Report analogue input or generate a sine wave for testing.
    // - We use a timer interrupt for accuracy of sampling, not accuracy of
    //   serial transmission.
    // - No need for ATOMIC_BLOCK because we're in the interrupt.

    if (!continuous_sampling && samples_to_go == 0) {
        return;
    }
    if (sample_pending) {
        missed_sample = true;
    }
    sample_time = micros();
    if (source == SOURCE_ECG) {
        sample_value = analogRead(ECG_PIN);  // range [0, 1023]
    } else if (source == SOURCE_SINE) {
        sine_time_s = fmod(sine_time_s + sampling_resolution_s, sine_phase_duration_s);
        float phase_radians = TWO_PI * (sine_time_s / sine_phase_duration_s);
        //                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        //                                    range 0-1
        //                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        //                            range 0 to 2*pi

        sample_value = HALF_OUTPUT + HALF_OUTPUT * sin(phase_radians);
        //             ^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^
        //             scale to output range       -1 to +1
    } else if (source == SOURCE_TIME) {
        sample_value = sample_time;
    }
    sample_pending = true;
}

void timerInterruptServiceRequest()
{
    ++timer_count;
    if (timer_count >= timer_units && !timer_overflowing) {
		timer_overflowing = true;
		timer_count -= timer_units;
        // ... subtract timer_units to catch missed overflows;
        // set to 0 if you don't want this.
		timerISRMain();
		timer_overflowing = false;
	}

}

void startTimer(float frequency_hz)
{
    Timer1.detachInterrupt();

    sampling_freq_hz = frequency_hz;
    sampling_resolution_s = 1.0 / frequency_hz;
    timer_units = 1;
    double timer_resolution_s = sampling_resolution_s;
    while (timer_resolution_s > MAX_TIMER_RESOLUTION_S) {
        // The timer won't time very long intervals. So notch down the
        // resolution and just count more.
        timer_units *= 2;
        timer_resolution_s /= 2;
    }
    unsigned long timer_period_microsec = timer_resolution_s * 1000000;
    if (debug_level >= 2) {
        Serial.print(strDebug);
        Serial.print("Starting timer: frequency (Hz) = ");
        Serial.print(frequency_hz, NUM_DP_HZ);
        Serial.print(", timer_units = ");
        Serial.print(timer_units);
        Serial.print(", timer resolution (s) = ");
        Serial.print(timer_resolution_s, NUM_DP_S_FOR_MICROSEC);
        Serial.print(", timer period (microseconds) = ");
        Serial.println(timer_period_microsec);
    }

    // For timer/ISR code, see https://github.com/TDeagan/pyEKGduino
    /*
    noInterrupts();  // Disable all interrupts before initialization
    FlexiTimer2::stop();  // in case it was running
    FlexiTimer2::set(units, timer_resolution, timerInterruptServiceRequest);
    FlexiTimer2::start();
    interrupts();  // Enable all interrupts after initialization has been completed
    */

    if (timer1_initialized) {
        Timer1.setPeriod(timer_period_microsec);
    } else {
        // First time
        Timer1.initialize(timer_period_microsec);
        timer1_initialized = true;
    }
    timer_count = 0;
    timer_overflowing = false;

    Timer1.attachInterrupt(timerInterruptServiceRequest);
}


//=============================================================================
// Parser
//=============================================================================

void processCommand()
{
    if (debug_level >= 1) {
        Serial.print(strDebug);
        Serial.print(strCommandReceived);
        Serial.println(incoming);
    }
    char* p = incoming;
    getNextWord(p, nextword);
    if (!*nextword) {
        if (debug_level >= 2) {
            debug(strBlankCommand);
        }
        return;
    }
    if (!strcmp(nextword, cmdSourceAnalogue)) {  // ECG source
        if (redundantParameters(p)) return;
        source = SOURCE_ECG;
    } else if (!strcmp(nextword, cmdSourceSineGenerator)) {  // sine wave generator source
        if (redundantParameters(p)) return;
        source = SOURCE_SINE;
    } else if (!strcmp(nextword, cmdSourceTime)) {  // sine wave generator source
        if (redundantParameters(p)) return;
        source = SOURCE_TIME;
    } else if (!strcmp(nextword, cmdSineGeneratorFrequency)) {  // set sine wave frequency
        float freq;
        if (!getNextFloat(p, freq)) return;
        if (redundantParameters(p)) return;
        setSineFrequency(freq);
    } else if (!strcmp(nextword, cmdSampleFrequency)) {  // set sampling frequency
        float freq;
        if (!getNextFloat(p, freq)) return;
        if (redundantParameters(p)) return;
        startTimer(freq);
    } else if (!strcmp(nextword, cmdSampleContinuous)) {  // continuous sampling
        if (redundantParameters(p)) return;
        setSampling(true, 0);
    } else if (!strcmp(nextword, cmdSampleNumber)) {  // request a specific number of samples
        int num_samples;
        if (!getNextInt(p, num_samples)) return;
        if (redundantParameters(p)) return;
        setSampling(false, num_samples);
    } else if (!strcmp(nextword, cmdQuiet)) {  // stop sending samples
        if (redundantParameters(p)) return;
        setSampling(false, 0);
    } else {
        error(strUnknownCommand);
        return;
    }
    Serial.println(respOK);
}


//=============================================================================
// Arduino main loop
//=============================================================================

void loop()
{
    // Deal with commands. Check if serial data available.
    if (Serial.available() > 0) {
        char c = (char) Serial.read();  // Read byte from serial port
        if (c == '\n' || c == '\r') {
            processCommand();
            resetInputBuffers();
        } else if (c < 32 || c > 127) {
            // nonprintable ASCII; ignore
            // (or possibly -1, meaning the board lied about availability)
        } else if (incomingIndex >= INPUT_BUFSIZE - 1) {
            error(strInputBufferOverflow);
            resetInputBuffers();
        } else {
            incoming[incomingIndex++] = c;
            incoming[incomingIndex] = 0;  // keep it null-terminated
        }
    }

    // Send outbound data. Currently UNBUFFERED.
    if (sample_pending) {
        unsigned long time;
        unsigned long value;
        bool finished_sample_run = false;
        ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
            time = sample_time;
            value = sample_value;
            if (!continuous_sampling && samples_to_go > 0) {
                --samples_to_go;
                if (samples_to_go == 0) {
                    finished_sample_run = true;
                }
            }
            sample_pending = false;
        }
        sendValue(time, value);
        if (missed_sample) {
            error(strSampleOverflow);
            missed_sample = false;
        }
        if (finished_sample_run && debug_level >= 2) {
            Serial.print(strDebug);
            Serial.print("Finished sampling; end time = ");
            Serial.println(micros());
        }
    }

}


//=============================================================================
// Arduino setup
//=============================================================================

void setup()
{
    // Hardware config
    pinMode(ECG_PIN, INPUT);
    // ... analogue pins can be configured as digital
    //     https://www.arduino.cc/en/Tutorial/DigitalPins
    analogReference(VOLTAGE_RANGE);
    // ... sets the voltage used as the top of the input range
    //     https://www.arduino.cc/reference/en/language/functions/analog-io/analogreference/
    // ... and the DFRobot sensor provides an output of 0 to 3.3 V:
    //     https://www.dfrobot.com/wiki/index.php/Heart_Rate_Monitor_Sensor_SKU:_SEN0213

    // Starting values
#ifdef DEBUG
    debug_level = 2;
#else
    debug_level = 0;
#endif
    source = SOURCE_ECG;
    setSineFrequency(DEFAULT_SINE_FREQUENCY_HZ);
    setSampling(false, 0);

    // Start the sampling timer
    timer1_initialized = false;
    startTimer(DEFAULT_SAMPLING_FREQUENCY_HZ);

    // Start serial comms
    Serial.begin(BAUD_RATE, SERIAL_DATA_PARITY_STOP);
    // Let's be very sure that our input buffer starts sensibly.
    resetInputBuffers();
    Serial.println(respHello);
    info(strVersion);
    info(strHardwareInfo);
}
