#include <Servo.h>


Servo servo_az;
Servo servo_el;

void setup() {
  Serial.begin(9600);
  
  servo_az.attach(9);  // Attach the servo to pin 9
  servo_el.attach(10); // Attach the servo to pin 10

  servo_az.write(65);
  servo_el.write(25);
}

void loop() {
  if (Serial.available() > 0) {
    
    char command = (char)Serial.read();

    if (command == 'A') {
      int pin = Serial.parseInt();
      float ang = Serial.parseFloat(SKIP_ALL);
    
      if (pin == 9)
      {
        servo_az.write(ang);
      }
      if (pin == 10)
      {
        servo_el.write(ang);
      }
    } 
    else if (command == 'B'){
      servo_az.write(65.00);
      servo_el.write(25.00);
    }

  }
}
