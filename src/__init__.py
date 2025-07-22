from os.path import abspath, dirname
import os


# Check if project root was set by parent process
if 'TPN3_PROJECT_ROOT' in os.environ:
    ROOT_PATH = os.environ['TPN3_PROJECT_ROOT']
else:
    ROOT_PATH = dirname(dirname(abspath(__file__)))
