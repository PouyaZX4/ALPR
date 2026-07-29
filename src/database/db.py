import sqlite3

class ViolationDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        pass

    def insert(self, record: dict):
        """
        Inserts a speeding violation record into SQLite DB.
        """
        pass

    def get_all(self) -> list[dict]:
        """
        Retrieves all violation records.
        """
        pass

    def get_recent(self, limit: int = 50) -> list[dict]:
        """
        Retrieves recent violation records up to specified limit.
        """
        pass
