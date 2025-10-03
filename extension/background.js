// Minimal background service worker for YouTube title event storage (title-only pipeline)
function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open('local-logger', 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains('events')) {
        const store = db.createObjectStore('events', { keyPath: 'id', autoIncrement: true });
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
      if (cur && out.length < limit) { out.push(cur.value); cur.continue(); } else resolve(out);
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

// Only keep whitelisted types originating from YouTube harvesting
const ALLOWED_TYPES = new Set(['feed_video', 'shorts_video']);
function sanitizeEvent(p) {
  try {
    if (!p || !ALLOWED_TYPES.has(p.type)) return null;
    const title = (p.title || '').trim();
    if (!title) return null;
    return {
      ts: p.ts || Date.now(),
      type: p.type,
      title,
      videoId: p.videoId,
      href: p.href,
      page: p.page,
      platform: p.platform || 'youtube'
    };
  } catch { return null; }
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  (async () => {
    try {
      if (msg && msg.type === 'LCL_ADD_EVENT') {
        const compact = sanitizeEvent(msg.payload);
        if (compact) {
          await addEvent(compact);
          sendResponse({ ok: true, stored: true });
        } else {
          sendResponse({ ok: true, stored: false });
        }
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
