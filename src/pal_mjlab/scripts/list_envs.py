"""Script to list MJLab environments."""

import pal_mjlab.tasks  # noqa: F401 to register environments
from mjlab.scripts.list_envs import main

if __name__ == "__main__":
    main()
