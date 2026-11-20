from pathlib import Path

PAL_MJLAB_SRC_PATH: Path = Path(__file__).parent

# Load mjlab before any other pal_mjlab module. Importing mjlab imports every registered task
# package (pal_mjlab.tasks included), and the tasks import pal_mjlab.robots and
# pal_mjlab.rl: if one of those were being imported first, its own first mjlab import would
# re-enter it half-initialized. With mjlab loaded here, the tasks always load first.
import mjlab  # noqa: E402, F401
