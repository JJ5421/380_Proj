from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import AngularServo
from time import sleep

factory = PiGPIOFactory()
joint_base = AngularServo(13, min_angle=-90, max_angle=90, pin_factory=factory)
angle_base = 0
while True:
    angle = int(input("Angle: "))

    angle_base = angle
    joint_base.angle = angle_base

    print(angle, angle_base)
