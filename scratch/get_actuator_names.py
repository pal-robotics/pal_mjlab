from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import KANGAROO_ACTUATOR_NAMES
print("--- START ACTUATOR LIST ---")
for i, name in enumerate(KANGAROO_ACTUATOR_NAMES):
    print(f"{i}: {name}")
print("--- END ACTUATOR LIST ---")
