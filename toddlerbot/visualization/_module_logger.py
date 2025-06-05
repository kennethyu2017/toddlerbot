import logging
logger=logging.getLogger(__name__)
# use the parent logger's level as effective-level.
# logger.setLevel(logging.DEBUG)
logger.propagate=True  # let the main.py to set it in order to stop cvae logger output.