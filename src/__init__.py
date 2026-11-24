"""Highway driving RL project package.

highway-env draws the road with pygame. Colab and other servers have no
screen, so we tell pygame to use a "dummy" (invisible) display. This must
happen before pygame is imported, so it lives in the package __init__.
"""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
