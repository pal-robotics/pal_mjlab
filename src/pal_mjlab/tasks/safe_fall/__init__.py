"""Safe-fall task: reach a safe, stationary, low state as fast as possible.

Fallback behavior for when a fall or an actuator fault is detected. Inspired by
Ha & Liu, "Multiple contact planning for minimizing damage of humanoid falls"
(IROS 2015): minimize the largest single impact rather than the total.
"""
