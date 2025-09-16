export function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open("local-logger", 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains("events")) {
        const store = db.createObjectStore("events", { keyPath: "id", autoIncrement: true });
        store.createIndex("by_url", "url");
        store.createIndex("by_ts", "ts");
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
export async function addEvent(evt) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("events", "readwrite");
    tx.objectStore("events").add(evt);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}
export async function getRecent(limit = 200) {
  const db = await openDB();
  return new Promise((resolve) => {
    const out = [];
    const tx = db.transaction("events", "readonly");
    const idx = tx.objectStore("events").index("by_ts");
    const req = idx.openCursor(null, "prev");
    req.onsuccess = e => {
      const cur = e.target.result;
      if (cur && out.length < limit) { out.push(cur.value); cur.continue(); }
      else resolve(out);
    };
  });
}
export async function clearAll() {
  const db = await openDB();
  return new Promise((resolve) => {
    const tx = db.transaction("events", "readwrite");
    tx.objectStore("events").clear();
    tx.oncomplete = () => resolve();
  });
}
