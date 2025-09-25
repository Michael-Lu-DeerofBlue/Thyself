// Background service worker (module) handles IndexedDB so content scripts avoid dynamic import issues.

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open('local-logger', 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains('events')) {
        const store = db.createObjectStore('events', { keyPath: 'id', autoIncrement: true });
        store.createIndex('by_url', 'url');
        store.createIndex('by_ts', 'ts');
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function addEvent(evt) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('events', 'readwrite');
    tx.objectStore('events').add(evt);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function getRecent(limit = 200) {
  const db = await openDB();
  return new Promise((resolve) => {
    const out = [];
    const tx = db.transaction('events', 'readonly');
    const idx = tx.objectStore('events').index('by_ts');
    const req = idx.openCursor(null, 'prev');
    req.onsuccess = e => {
      const cur = e.target.result;
      if (cur && out.length < limit) { out.push(cur.value); cur.continue(); }
      else resolve(out);
    };
  });
}

async function clearAll() {
  const db = await openDB();
  return new Promise((resolve) => {
    const tx = db.transaction('events', 'readwrite');
    tx.objectStore('events').clear();
    tx.oncomplete = () => resolve();
  });
}

chrome.runtime.onInstalled.addListener(() => {
  console.log('Local Content Logger installed (background active).');
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  (async () => {
    try {
      if (msg && msg.type === 'LCL_ADD_EVENT') {
        await addEvent(msg.payload);
        sendResponse({ ok: true });
      } else if (msg && msg.type === 'LCL_GET_RECENT') {
        const data = await getRecent(msg.limit || 200);
        sendResponse({ ok: true, data });
      } else if (msg && msg.type === 'LCL_CLEAR') {
        await clearAll();
        sendResponse({ ok: true });
      }
    } catch (e) {
      sendResponse({ ok: false, error: String(e) });
    }
  })();
  return true; // keep message channel open for async
});
