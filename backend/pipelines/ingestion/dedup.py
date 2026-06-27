import hashlib
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'db', 'ingestion.sqlite')

def _get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            content_hash TEXT,
            status TEXT
        )
    ''')
    conn.commit()
    return conn

def compute_sha256(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

def check_duplicate(content_hash: str) -> str:
    """Returns document_id if found, else None"""
    conn = _get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM documents WHERE content_hash = ?', (content_hash,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def record_document(doc_id: str, content_hash: str, status: str = 'queued'):
    conn = _get_db()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT OR REPLACE INTO documents (id, content_hash, status) VALUES (?, ?, ?)',
        (doc_id, content_hash, status)
    )
    conn.commit()
    conn.close()

def update_document_status(doc_id: str, status: str):
    conn = _get_db()
    cursor = conn.cursor()
    cursor.execute('UPDATE documents SET status = ? WHERE id = ?', (status, doc_id))
    conn.commit()
    conn.close()
