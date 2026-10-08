// idb storage for transferring decrypted datasets to lab on same origin

const DB_NAME = 'bioblock-transfer';
const STORE_NAME = 'datasets';
const DB_VERSION = 1;
const TTL = 30 * 60 * 1000;

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'cid' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function storeForTransfer(
  cid: string,
  blob: Blob,
  fileName: string
): Promise<void> {
  if (typeof indexedDB === 'undefined') return;
  const db = await openDB();
  const tx = db.transaction(STORE_NAME, 'readwrite');
  const store = tx.objectStore(STORE_NAME);

  const allKeys = await idbReq<IDBValidKey[]>(store.getAllKeys());
  for (const key of allKeys) {
    const item = await idbReq<TransferRecord | undefined>(store.get(key));
    if (item && Date.now() - item.ts > TTL) {
      store.delete(key);
    }
  }

  store.put({ cid, blob, fileName, ts: Date.now() });
  await idbTx(tx);
  db.close();
}

interface TransferRecord {
  cid: string;
  blob: Blob;
  fileName: string;
  ts: number;
}

function idbReq<T>(req: IDBRequest): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result as T);
    req.onerror = () => reject(req.error);
  });
}

function idbTx(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}
