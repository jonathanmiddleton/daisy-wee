import logging.config
import yaml, os


class MasterLogger:
    _logger = None
    _is_master = None
    _initialized = False

    @classmethod
    def _initialize(cls):
        if not cls._initialized:
            cls._is_master = int(os.getenv("RANK", 0)) == 0
            if cls._is_master:
                with open("config/logging.yml", "r") as f:
                    config = yaml.safe_load(f.read())
                    logging.config.dictConfig(config)
            cls._logger = logging.getLogger(__name__)
            cls._initialized = True

    @classmethod
    def debug(cls, st):
        cls._initialize()
        if cls._is_master:
            cls._logger.debug(st)

    @classmethod
    def info(cls, st):
        cls._initialize()
        if cls._is_master:
            cls._logger.info(st)

    @classmethod
    def warning(cls, st):
        cls._initialize()
        if cls._is_master:
            cls._logger.warning(st)

    @classmethod
    def error(cls, st):
        cls._initialize()
        if cls._is_master:
            cls._logger.error(st)