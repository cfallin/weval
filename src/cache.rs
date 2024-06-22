//! Caching of weval results.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

pub type ModuleHash = [u8; 32]; // SHA-256 hash.

pub fn compute_hash(raw_bytes: &[u8]) -> ModuleHash {
    Sha256::digest(raw_bytes).into()
}

/// Cache result: compiled Wasm bytecode, with signature.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheData {
    /// Function signature index.
    pub sig: u32,
    /// Function name.
    pub name: String,
    /// Raw function body bytecode, including locals.
    pub body: Vec<u8>,
}

pub struct Cache {
    module_hash: ModuleHash,
    db: sqlite::ConnectionThreadSafe,
}

pub struct CacheThreadCtx<'a> {
    cache: &'a Cache,
    lookup_stmt: sqlite::Statement<'a>,
    insert_stmt: sqlite::Statement<'a>,
}

impl Cache {
    pub fn open(path: &Path, module_hash: ModuleHash) -> anyhow::Result<Cache> {
        let db = sqlite::Connection::open_thread_safe(path)?;
        db.execute(
            r#"
            CREATE TABLE IF NOT EXISTS weval_cache(
                module_hash BLOB NOT NULL,
                key BLOB NOT NULL,
                result BLOB NOT NULL,
                created_time INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx ON weval_cache(
                 module_hash, key
             );
        "#,
        )?;
        Ok(Cache { module_hash, db })
    }

    pub fn thread(&self) -> anyhow::Result<CacheThreadCtx<'_>> {
        let lookup_stmt = self.db.prepare(
            r#"
            SELECT result FROM weval_cache
                WHERE module_hash=? AND key=?
            "#,
        )?;
        let insert_stmt = self.db.prepare(
            r#"
            INSERT INTO weval_cache
                (module_hash, key, result, created_time)
            VALUES
                (?, ?, ?, unixepoch())
            "#,
        )?;
        Ok(CacheThreadCtx {
            cache: self,
            lookup_stmt,
            insert_stmt,
        })
    }
}

impl<'a> CacheThreadCtx<'a> {
    pub fn lookup(&mut self, key: &[u8]) -> anyhow::Result<Option<CacheData>> {
        self.lookup_stmt.bind((1, &self.cache.module_hash[..]))?;
        self.lookup_stmt.bind((2, key))?;

        let mut result = None;
        while self.lookup_stmt.next()? == sqlite::State::Row {
            let data: Vec<u8> = self.lookup_stmt.read(0)?;
            result = Some(bincode::deserialize(&data)?);
        }

        self.lookup_stmt.reset()?;
        Ok(result)
    }

    pub fn insert(&mut self, key: &[u8], data: CacheData) -> anyhow::Result<()> {
        let data = bincode::serialize(&data)?;
        self.insert_stmt.bind((1, &self.cache.module_hash[..]))?;
        self.insert_stmt.bind((2, key))?;
        self.insert_stmt.bind((3, &data[..]))?;
        while self.insert_stmt.next()? == sqlite::State::Row {}
        self.insert_stmt.reset()?;
        Ok(())
    }
}
