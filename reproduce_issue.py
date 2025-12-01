import sys
import os

# Add src and app to path as the notebook does
sys.path.append(os.path.abspath('src'))
sys.path.append(os.path.abspath('app'))

try:
    print("Attempting to import envs.navigation_2d...")
    from envs.navigation_2d import Navigation2DEnv
    print("Successfully imported Navigation2DEnv")
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

try:
    print("Attempting to import moviepy...")
    import moviepy
    print(f"Moviepy version: {moviepy.__version__}")
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    print("Successfully imported ImageSequenceClip")
except ImportError as e:
    print(f"Moviepy import failed: {e}")
