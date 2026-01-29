#!/usr/bin/env python3
# =====================================
# Author : Adnan Abdullah
# Email: adnanabdullah@ufl.edu
# =====================================

"""
Keyboard tele-operation interface for cavepi.
 - Manual planar movement with WASD-like keys
 - Vertical motion (“heave”) is overridden by /cavepi/heave_control_input unless
   the operator presses i (up) or k (down)
 - Deadman: if no key press is seen for ~deadman_timeout_s, planar commands go to 0
"""

from __future__ import print_function

import sys
import select
import termios
import tty
import time

import rospy
from std_msgs.msg import Float32, Bool
from cavepi_interfaces.msg import cavepiInput

BANNER = """
Reading from the keyboard
---------------------------
Planar Movement:
   q    w    e
   a         d
   z    x    c

i : up  (+z)
k : down (-z)

Anything else : stop

r / v : increase / decrease thruster power by 10 %%
---------------------------
CTRL-C to quit
"""

# (surge, sway, heave, roll, pitch, yaw)
MOVE_BINDINGS = {
    'q': ( (2**0.5)/2, 0, 0, 0, 0, -(2**0.5)/2),
    'w': ( 1,          0, 0, 0, 0,  0),
    'e': ( (2**0.5)/2, 0, 0, 0, 0,  (2**0.5)/2),
    'a': ( 0,          0, 0, 0, 0, -1),
    'd': ( 0,          0, 0, 0, 0,  1),
    'z': (-(2**0.5)/2, 0, 0, 0, 0, -(2**0.5)/2),
    'x': (-1,          0, 0, 0, 0,  0),
    'c': (-(2**0.5)/2, 0, 0, 0, 0,  (2**0.5)/2),

    'i': ( 0, 0, -1, 0, 0, 0),   # ascend  (note: sign matches your earlier convention)
    'k': ( 0, 0,  1, 0, 0, 0),   # descend
}

SPEED_BINDINGS = {
    'r': +0.10,     # faster
    'v': -0.10,     # slower
}


class cavepiTeleop:
    def __init__(self):
        self._term_settings = termios.tcgetattr(sys.stdin)

        self.cmd_pub = rospy.Publisher("/cavepi/user_input", cavepiInput, queue_size=1)
        self.ind_pub = rospy.Publisher("/cavepi/depth_change_indicator", Bool, queue_size=1)

        rospy.Subscriber("/cavepi/heave_control_input", Float32, self._heave_control_cb, queue_size=1)

        # Speed scaling (0..1)
        self.speed = rospy.get_param("~speed", 0.05)

        # Deadman timeout: if no key for this long, planar motion is zeroed
        self.deadman_timeout_s = rospy.get_param("~deadman_timeout_s", 0.25)

        # Manual heave hold: i/k must be refreshed within this window to stay active
        self.manual_heave_hold_s = rospy.get_param("~manual_heave_hold_s", 0.15)

        # Current command state (unitless, -1..1 before speed scaling)
        self.surge = 0.0
        self.roll  = 0.0
        self.yaw   = 0.0

        # Heave control: either manual (-1..1) briefly, else controlled_heave from controller
        self.controlled_heave = 0.0
        self.manual_heave = 0.0
        self._manual_heave_active_until = 0.0

        self._last_key_time = time.time()

        rospy.loginfo("Thruster power initialised to %.0f %%", self.speed * 100)

    # ---------- Terminal key IO ----------
    def _get_key(self, timeout=0.05):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        key = sys.stdin.read(1) if rlist else ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._term_settings)
        return key

    # ---------- Subscriber ----------
    def _heave_control_cb(self, msg: Float32):
        self.controlled_heave = float(msg.data)

    # ---------- Publishing ----------
    def _publish_cmd(self):
        now = time.time()

        # Deadman: if no recent key activity, stop planar commands
        if (now - self._last_key_time) > self.deadman_timeout_s:
            self.surge = 0.0
            self.roll  = 0.0
            self.yaw   = 0.0

        # Manual heave active?
        manual_active = now <= self._manual_heave_active_until
        depth_changed = manual_active  # for the indicator topic

        heave_to_send = self.manual_heave if manual_active else self.controlled_heave

        cmd = cavepiInput()
        cmd.surge = float(self.surge * self.speed)
        cmd.roll  = float(self.roll  * self.speed)
        cmd.yaw   = float(self.yaw   * self.speed)

        # NOTE: we do NOT multiply controller output by speed unless you really intend that.
        # Usually /cavepi/heave_control_input is already in [-1,1] command units.
        # If you want scaling, keep "* self.speed" here; otherwise remove it.
        cmd.heave = float(heave_to_send * self.speed)

        self.cmd_pub.publish(cmd)
        self.ind_pub.publish(Bool(data=depth_changed))

    def _stop_all(self):
        self.surge = 0.0
        self.roll  = 0.0
        self.yaw   = 0.0
        self.manual_heave = 0.0
        self._manual_heave_active_until = 0.0
        self._publish_cmd()

    # ---------- Main loop ----------
    def run(self):
        print(BANNER)
        rate = rospy.Rate(20)
        try:
            while not rospy.is_shutdown():
                key = self._get_key(timeout=0.05)

                if key:
                    self._last_key_time = time.time()

                # Movement keys
                if key in MOVE_BINDINGS:
                    surge, _, heave, roll, _, yaw = MOVE_BINDINGS[key]
                    self.surge = float(surge)
                    self.roll  = float(roll)
                    self.yaw   = float(yaw)

                    # Manual heave is *momentary* (must be refreshed)
                    if key in ('i', 'k'):
                        self.manual_heave = float(heave)
                        self._manual_heave_active_until = time.time() + self.manual_heave_hold_s

                # Speed adjustment
                elif key in SPEED_BINDINGS:
                    self.speed = max(0.0, min(1.0, self.speed + SPEED_BINDINGS[key]))
                    rospy.loginfo("Thruster power now %.0f %%", self.speed * 100)

                # Graceful quit on CTRL-C
                elif key == '\x03':
                    self._stop_all()
                    break

                # Any other key: immediate stop (but heave returns to controller)
                elif key != '':
                    self.surge = 0.0
                    self.roll  = 0.0
                    self.yaw   = 0.0
                    self.manual_heave = 0.0
                    self._manual_heave_active_until = 0.0

                # Publish every cycle
                self._publish_cmd()
                rate.sleep()

        finally:
            self._stop_all()
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._term_settings)


if __name__ == '__main__':
    rospy.init_node('cavepi_teleop_keyboard')
    cavepiTeleop().run()
