/*
This file is developed based on the official arduino example code for Portenta
https://docs.arduino.cc/tutorials/portenta-h7-lite/reading-writing-flash-memory

Here, we only want to measure the time/energy overhead of reading 64KB of data from the Flash
We put a high on a pin and then start the reading, and put a low on the pin when it finishes
We use Discovery Ananlog to measure the time and energy overhead for the high duration 
*/

#include <FlashIAPBlockDevice.h>
#include "FlashIAPLimits.h"

using namespace mbed;

void setup() {
  // put your setup code here, to run once:

  auto [_, startAddress, iapSize] = getFlashIAPLimits();
  FlashIAPBlockDevice blockDevice(startAddress, iapSize);
  blockDevice.init();


  const auto dataSize = 1024; // 1KB
  const auto iteration = 640; // 64KB
  char buffer[dataSize] {};

  delay(500);
  digitalWrite(15, HIGH);
  for(int iter = 0; iter < iteration; iter++){
    blockDevice.read(buffer, 0, dataSize);  // read 1KB data in one iteration
  }
  blockDevice.deinit();
  digitalWrite(15, LOW);

}

void loop() {
  // put your main code here, to run repeatedly:

}
