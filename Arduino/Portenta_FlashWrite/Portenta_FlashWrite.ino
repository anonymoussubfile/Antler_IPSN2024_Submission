'''
This file is developed based on the official arduino example code for Portenta
https://docs.arduino.cc/tutorials/portenta-h7-lite/reading-writing-flash-memory

Here, we only want to measure the time/energy overhead of reading 64KB of data from the Flash
We put a high on a pin and then start the reading, and put a low on the pin when it finishes
We use Discovery Ananlog to measure the time and energy overhead for the high duration 
'''

#include <FlashIAPBlockDevice.h>
#include "FlashIAPLimits.h"

using namespace mbed;

void setup() {
  // put your setup code here, to run once:

  Serial.begin(115200);
  while (!Serial);

  Serial.println("\n\nFlashIAPBlockDevice Test");
  Serial.println("------------------------");  

  // Feed the random number generator for later content generation
  randomSeed(analogRead(0));

  // Get limits of the the internal flash of the microcontroller
  auto [flashSize, startAddress, iapSize] = getFlashIAPLimits();

  Serial.print("Flash Size: ");
  Serial.print(flashSize / 1024.0 / 1024.0);
  Serial.println(" MB");
  Serial.print("FlashIAP Start Address: 0x");
  Serial.println(startAddress, HEX);
  Serial.print("FlashIAP Size: ");
  Serial.print(iapSize / 1024.0 / 1024.0);
  Serial.println(" MB");

  // Create a block device on the available space of the flash
  FlashIAPBlockDevice blockDevice(startAddress, iapSize);
  
  // Initialize the Flash IAP block device and print the memory layout
  blockDevice.init();

  const auto eraseBlockSize = blockDevice.get_erase_size();
  const auto programBlockSize = blockDevice.get_program_size();

  Serial.println("Block device size: " + String((unsigned int) blockDevice.size() / 1024.0 / 1024.0) + " MB");
  Serial.println("Readable block size: " + String((unsigned int) blockDevice.get_read_size())  + " bytes");
  Serial.println("Programmable block size: " + String((unsigned int) programBlockSize) + " bytes");
  Serial.println("Erasable block size: " + String((unsigned int) eraseBlockSize / 1024) + " KB");
      
  // String newMessage = "Random number: " + String(random(1024));
  int messages[10] = {1,2,3,4,5,6,7,8,9,10};
  String newMessage = "";
  for(int i = 0; i < 10; i++){
    newMessage += String(messages[i]);
  }


  // Calculate the amount of bytes needed to store the message
  // This has to be a multiple of the program block size
  const auto messageSize = newMessage.length() + 1; // C String takes 1 byte for NULL termination
  const unsigned int requiredEraseBlocks = ceil(messageSize / (float)  eraseBlockSize);
  const unsigned int requiredProgramBlocks = ceil(messageSize / (float)  programBlockSize);
  const auto dataSize = requiredProgramBlocks * programBlockSize;  
  char buffer[dataSize] {};

  // Read back what was stored at previous execution
  Serial.println("Reading previous message...");
  blockDevice.read(buffer, 0, dataSize);
  Serial.println(buffer);

  // Erase a block starting at the offset 0 relative
  // to the block device start address
  blockDevice.erase(0, requiredEraseBlocks * eraseBlockSize);

  // Write an updated message to the first block
  Serial.println("Writing new message...");
  Serial.println(newMessage);  
  blockDevice.program(newMessage.c_str(), 0, dataSize);

  // Deinitialize the device
  blockDevice.deinit();
  Serial.println("Done.");

}

void loop() {
  // put your main code here, to run repeatedly:

}
