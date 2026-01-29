#!/usr/bin/env python3

# =====================================
# ROV joystick teleop for cavepiInput
# Author: Adnan / Alankrit
# =====================================

import rospy
from sensor_msgs.msg import Joy
from cavepi_interfaces.msg import cavepiInput


class cavepiJoyTeleop(object):
    def __init__(self):
        rospy.init_node('cavepi_teleop_joy')

        # Publish to same topic as keyboard teleop
        self.pub = rospy.Publisher("/cavepi/user_input",
                                   cavepiInput,
                                   queue_size=1)

        # Parameters
        # Base speed scalar (like your ~speed param)
        self.speed = rospy.get_param("~speed", 0.3)    # 0..1
        self.speed_step = rospy.get_param("~speed_step", 0.1)
        self.speed_min = rospy.get_param("~speed_min", 0.0)
        self.speed_max = rospy.get_param("~speed_max", 1.0)

        # Joystick mapping (adjust for your controller)
        # Default Logitech/Xbox style:
        # axes[0] = left stick LR (-1 left, +1 right)
        # axes[1] = left stick UD (+1 up, -1 down)
        # axes[3] = right stick LR
        # axes[4] = right stick UD
        #
        # Example mapping:
        #   surge -> left stick vertical (axes[1])
        #   yaw   -> left stick horizontal (axes[0])
        #   heave -> right stick vertical (axes[4])
        #   roll  -> right stick horizontal (axes[3])
        self.axis_surge = rospy.get_param("~axis_surge", 1)
        self.axis_yaw   = rospy.get_param("~axis_yaw",   0)
        self.axis_heave = rospy.get_param("~axis_heave", 4)
        self.axis_roll  = rospy.get_param("~axis_roll",  3)

        # Buttons for speed up/down
        # (Change these indices for your joystick)
        # Example: button 4 (LB) decreases, button 5 (RB) increases
        self.btn_speed_down = rospy.get_param("~btn_speed_down", 4)
        self.btn_speed_up   = rospy.get_param("~btn_speed_up",   5)

        # Deadzone
        self.deadzone = rospy.get_param("~deadzone", 0.05)

        # Last Joy message (for continuous publish)
        self.last_joy = None

        rospy.Subscriber("joy", Joy, self.joy_callback)

        # Publish at fixed rate even if joystick unchanged
        self.rate = rospy.Rate(20)  # 20 Hz
        self.spin()

    def joy_callback(self, msg):
        self.last_joy = msg

        # Handle speed scaling from buttons
        # Buttons are 1 when pressed
        if self.btn_speed_up < len(msg.buttons) and msg.buttons[self.btn_speed_up]:
            self.speed += self.speed_step
            if self.speed > self.speed_max:
                self.speed = self.speed_max
                rospy.loginfo("Thruster power at maximum.")
            else:
                rospy.loginfo("Thruster power: %.0f%%", self.speed * 100.0)

        if self.btn_speed_down < len(msg.buttons) and msg.buttons[self.btn_speed_down]:
            self.speed -= self.speed_step
            if self.speed < self.speed_min:
                self.speed = self.speed_min
                rospy.loginfo("Thruster power at minimum.")
            else:
                rospy.loginfo("Thruster power: %.0f%%", self.speed * 100.0)

    def apply_deadzone(self, v):
        return 0.0 if abs(v) < self.deadzone else v

    def spin(self):
        rospy.loginfo("cavepi joystick teleop running.")
        rospy.loginfo("Publishing cavepiInput to /cavepi/user_input")

        while not rospy.is_shutdown():
            if self.last_joy is not None:
                axes = self.last_joy.axes
                # Safe access with defaults
                def get_axis(idx):
                    return axes[idx] if 0 <= idx < len(axes) else 0.0

                # Raw joystick values in [-1, 1]
                surge_raw = self.apply_deadzone(get_axis(self.axis_surge))
                yaw_raw   = - self.apply_deadzone(get_axis(self.axis_yaw))
                heave_raw = self.apply_deadzone(get_axis(self.axis_heave))
                roll_raw  = - self.apply_deadzone(get_axis(self.axis_roll))

                # Same convention as your keyboard teleop:
                # scale by "speed"
                cmd = cavepiInput()
                cmd.surge = surge_raw * self.speed
                cmd.heave = heave_raw * self.speed  # heave is vertical, so divide by 3 to reduce
                cmd.roll  = roll_raw  * self.speed
                cmd.yaw   = yaw_raw   * self.speed

                self.pub.publish(cmd)

            self.rate.sleep()


if __name__ == "__main__":
    try:
        cavepiJoyTeleop()
    except rospy.ROSInterruptException:
        pass
