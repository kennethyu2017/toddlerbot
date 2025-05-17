from typing import Dict, Optional
import sys
import logging
from pathlib import Path

# TODO: use YAML -> dictConfig instead.
def config_logging(*, root_logger_level:int, root_handler_level:int,
                   root_fmt:str, root_date_fmt:str, log_file:Optional[str],
                   module_logger_config:Dict[str, int]):
    """
    1.we set the logging format and output handler in `root`, and shared among all the child loggers
        through the `propagate` mechanism.
    2.toggle the logging message of individual module through setting the corresponding logger`s level.

    Args:
        root_logger_level: severity for root logger. logging.INFO, logging.DEBUG,...
        root_handler_level: severity for root handler, effective control on child logger's propagating messages.
                            logging.INFO, logging.DEBUG,...
        root_fmt: comprehensive format string for root. only support `{` style.
        root_date_fmt: datetime format string for root. only support `{` style.
        log_file: destination of root handler. if None, we use StreamHandler to sys.stdout
        module_logger_config (dict): {logger_name:level, ...}

    Returns:

    """
    _config_root_logger(logger_level=root_logger_level,
                        handler_level=root_handler_level,
                        fmt=root_fmt,
                        date_fmt=root_date_fmt, log_file=log_file)
    _config_module_logger(module_logger_config)

def _config_root_logger(*, logger_level:int, handler_level: int,
                        fmt:str, date_fmt:str, log_file:Optional[str] = None):

    # configure root logger.
    root_lgr = logging.getLogger('root')

    # set level explicitly.
    root_lgr.setLevel(logger_level)

    # imported dm_control will add stream handler to `root` logger, we clear them first.
    if root_lgr.hasHandlers():
        print('---root logger handlers is not empty, we clear them first:')
        for _h in root_lgr.handlers:
            print(f'remove root logger handler:{_h.__str__()}')
            root_lgr.removeHandler(_h)

    assert not root_lgr.hasHandlers()

    # set handlers.
    if log_file is None:
        root_handler = logging.StreamHandler(stream=sys.stdout)
    else:
        file_path = Path(log_file)
        if not file_path.exists():
            if not file_path.parent.exists():
                file_path.parent.mkdir(parents=True)
            file_path.touch()
        root_handler = logging.FileHandler(filename=log_file, mode='a', encoding='UTF-8')
    # this handler is capable of outputting everything, but we control the output severity through
    # the corresponding logger's level.
    # root_handler.setLevel(logging.NOTSET)
    root_handler.setLevel(handler_level)
    root_formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt,style='{')
    root_handler.setFormatter(root_formatter)

    root_lgr.addHandler(root_handler)

# we don't specify handlers for modules, and just propagate to root handlers.
def _config_module_logger(module_configs:Dict[str, int]):
    # control module's logger. the individual logger will be created during `import` stage, before here.
    for _name, _level in module_configs.items():
        lgr = logging.getLogger(_name)
        lgr.setLevel(_level)
        # propagate to root's handler, not set the individual handler for module's logger.
        lgr.propagate=True
