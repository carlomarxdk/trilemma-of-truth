import sqlite3
from pathlib import Path
from datetime import datetime


class LogDataBase:
    '''
    A class to log experiment data into a SQLite database.
    '''
    def __init__(self, tab_name: str, db_name: str = 'experiments', path: str = 'outputs/'):
        self.tab_name = tab_name
        self.db_name = db_name
        # Build the full path to the database file
        self.db_path = Path(path) / f"{self.db_name}.db"
        self.connector = sqlite3.connect(str(self.db_path))
        self.cursor = self.connector.cursor()
        # Create a table with a proper schema for experiments
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {self.tab_name} (
                trial_id TEXT PRIMARY KEY,
                model TEXT,
                datapack TEXT,
                task TEXT,
                parameters TEXT,
                progress REAL,
                status INTEGER,
                date TEXT,
                time TEXT
            )
        ''')
        self.connector.commit()

    def write(self, trial_id: str, model: str, datapack: str, task: str, parameters: str, progress: float, status: int):
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")  # Date in YYYY-MM-DD format
        current_time = now.strftime("%H:%M")     # Time in HH:MM format
        query = f'''
            INSERT INTO {self.tab_name} (trial_id, model, datapack, task, parameters, progress, status, date, time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(trial_id) DO UPDATE SET
                model=excluded.model,
                datapack=excluded.datapack,
                task=excluded.task,
                parameters=excluded.parameters,
                progress=excluded.progress,
                status=excluded.status,
                date=excluded.date,
                time=excluded.time
        '''
        self.cursor.execute(query, (trial_id, model, datapack, task,
                            parameters, progress, status, current_date, current_time))
        self.connector.commit()
