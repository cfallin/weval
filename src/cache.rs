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
    db: Option<sqlite::ConnectionThreadSafe>,
    db_ro: Option<sqlite::ConnectionThreadSafe>,
}

pub struct CacheThreadCtx<'a> {
    cache: &'a Cache,
    lookup_stmt: Option<sqlite::Statement<'a>>,
    insert_stmt: Option<sqlite::Statement<'a>>,
    ro_lookup_stmt: Option<sqlite::Statement<'a>>,
}

impl Cache {
    pub fn open(
        path: Option<&Path>,
        path_ro: Option<&Path>,
        module_hash: ModuleHash,
    ) -> anyhow::Result<Cache> {
        let db = match path {
            Some(path) => {
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
                Some(db)
            }
            None => None,
        };
        let db_ro = match path_ro {
            Some(path_ro) => Some(sqlite::Connection::open_thread_safe_with_flags(
                path_ro,
                sqlite::OpenFlags::new().with_read_only(),
            )?),
            None => None,
        };
        Ok(Cache {
            module_hash,
            db,
            db_ro,
        })
    }

    pub fn can_insert(&self) -> bool {
        self.db.is_some()
    }

    pub fn thread(&self) -> anyhow::Result<CacheThreadCtx<'_>> {
        let lookup_stmt = match self.db.as_ref() {
            Some(db) => Some(db.prepare(
                r#"
                SELECT result FROM weval_cache
                    WHERE module_hash=? AND key=?
                "#,
            )?),
            None => None,
        };
        let ro_lookup_stmt = match self.db_ro.as_ref() {
            Some(db_ro) => Some(db_ro.prepare(
                r#"
                SELECT result FROM weval_cache
                    WHERE module_hash=? AND key=?
                "#,
            )?),
            None => None,
        };
        let insert_stmt = match self.db.as_ref() {
            Some(db) => Some(db.prepare(
                r#"
                INSERT INTO weval_cache
                    (module_hash, key, result, created_time)
                VALUES
                    (?, ?, ?, unixepoch())
                "#,
            )?),
            None => None,
        };
        Ok(CacheThreadCtx {
            cache: self,
            lookup_stmt,
            insert_stmt,
            ro_lookup_stmt,
        })
    }
}

impl<'a> CacheThreadCtx<'a> {
    pub fn lookup(&mut self, key: &[u8]) -> anyhow::Result<Option<CacheData>> {
        let mut result = None;
        for lookup in self
            .ro_lookup_stmt
            .iter_mut()
            .chain(self.lookup_stmt.iter_mut())
        {
            lookup.bind((1, &self.cache.module_hash[..]))?;
            lookup.bind((2, key))?;

            while lookup.next()? == sqlite::State::Row {
                let data: Vec<u8> = lookup.read(0)?;
                result = Some(bincode::deserialize(&data)?);
            }

            lookup.reset()?;
            if result.is_some() {
                break;
            }
        }
        Ok(result)
    }

    pub fn insert(&mut self, key: &[u8], data: CacheData) -> anyhow::Result<()> {
        if let Some(insert) = self.insert_stmt.as_mut() {
            let data = bincode::serialize(&data)?;
            insert.bind((1, &self.cache.module_hash[..]))?;
            insert.bind((2, key))?;
            insert.bind((3, &data[..]))?;
            while insert.next()? == sqlite::State::Row {}
            insert.reset()?;
        }
        Ok(())
    }
}
