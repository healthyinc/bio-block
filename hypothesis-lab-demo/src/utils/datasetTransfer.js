// helper to retrieve dataset passed from bio-block via idb or postMessage

const DB_NAME = 'bioblock-transfer';
const STORE_NAME = 'datasets';
const DB_VERSION = 1;

function getFromIdb(cid) {
  return new Promise((resolve) => {
    if (typeof indexedDB === 'undefined') return resolve(null);
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onerror = () => resolve(null);
    req.onsuccess = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.close();
        return resolve(null);
      }
      const tx = db.transaction(STORE_NAME, 'readonly');
      const store = tx.objectStore(STORE_NAME);
      const getReq = store.get(cid);
      getReq.onsuccess = () => {
        const item = getReq.result;
        db.close();
        if (item && item.blob) {
          resolve(new File([item.blob], item.fileName || `${cid}.csv`, { type: 'text/csv' }));
        } else {
          resolve(null);
        }
      };
      getReq.onerror = () => {
        db.close();
        resolve(null);
      };
    };
  });
}

function requestFromOpener(cid, timeoutMs = 4000) {
  return new Promise((resolve) => {
    if (!window.opener) {
      return resolve(null);
    }

    let timer = null;
    let pollInterval = null;

    const cleanup = () => {
      if (timer) clearTimeout(timer);
      if (pollInterval) clearInterval(pollInterval);
      window.removeEventListener('message', onMessage);
    };

    const onMessage = (e) => {
      if (e.data?.type === 'BIOBLOCK_DELIVER_DATASET' && e.data.cid === cid) {
        cleanup();
        const { blob, fileName } = e.data;
        if (blob) {
          const file = blob instanceof File
            ? blob
            : new File([blob], fileName || `${cid}.csv`, { type: 'text/csv' });
          resolve(file);
        } else {
          resolve(null);
        }
      }
    };

    window.addEventListener('message', onMessage);

    const ping = () => {
      try {
        window.opener.postMessage({ type: 'BIOBLOCK_REQUEST_DATASET', cid }, '*');
      } catch {
        cleanup();
        resolve(null);
      }
    };

    ping();
    pollInterval = setInterval(ping, 250);

    timer = setTimeout(() => {
      cleanup();
      resolve(null);
    }, timeoutMs);
  });
}

export async function loadDatasetForCid(cid) {
  if (!cid) return null;

  // try idb first if same origin
  const fromIdb = await getFromIdb(cid).catch(() => null);
  if (fromIdb) return fromIdb;

  // request from opener via postMessage
  if (window.opener) {
    return await requestFromOpener(cid);
  }

  return null;
}
