import sqlite3
import os

DB_PATH = 'football_celebration.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL
        )
    ''')
    
    # Create progress table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS progress (
            user_id INTEGER,
            celebration_code TEXT,
            PRIMARY KEY (user_id, celebration_code),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()

def get_users():
    """Return a list of dictionaries with user id and username."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT id, username FROM users ORDER BY id ASC')
    users = [{"id": row[0], "username": row[1]} for row in cursor.fetchall()]
    conn.close()
    return users

def add_user(username):
    """Add a new user and return their id. Return None if user exists."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO users (username) VALUES (?)', (username,))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        conn.close()
        return None

def get_user_progress(user_id):
    """Return a set of celebration_codes unlocked by the user."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT celebration_code FROM progress WHERE user_id = ?', (user_id,))
    codes = {row[0] for row in cursor.fetchall()}
    conn.close()
    return codes

def save_progress(user_id, celebration_code):
    """Save an unlocked celebration for a user."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR IGNORE INTO progress (user_id, celebration_code)
        VALUES (?, ?)
    ''', (user_id, celebration_code))
    conn.commit()
    conn.close()

def reset_progress(user_id):
    """Reset the progress for a specific user."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM progress WHERE user_id = ?', (user_id,))
    conn.commit()
    conn.close()

# Initialize the database file and tables on import
init_db()
