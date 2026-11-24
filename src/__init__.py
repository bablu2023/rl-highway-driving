"""Highway driving RL project package.

Rendering note: do NOT set SDL_VIDEODRIVER=dummy. highway-env turns its
drawing off when it sees that setting, and every video frame comes out black.
highway-env already renders off-screen (no display needed, works in Colab)
whenever the environment is created with render_mode="rgb_array".
"""
