from optuna.storages import RDBStorage
from optuna_dashboard import wsgi

storage = RDBStorage("postgresql://optuna_user:password@localhost/optuna_db")
application = wsgi(storage)