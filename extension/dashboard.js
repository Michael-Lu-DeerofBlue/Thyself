function getRecent(limit=200) {
  return new Promise(resolve => {
    chrome.runtime.sendMessage({ type: 'LCL_GET_RECENT', limit }, resp => {
      resolve((resp && resp.ok && resp.data) ? resp.data : []);
    });
  });
}
function clearAll() {
  return new Promise(resolve => {
    chrome.runtime.sendMessage({ type: 'LCL_CLEAR' }, () => resolve());
  });
}
async function render() {
  const items = await getRecent(200);
  document.getElementById("out").textContent = JSON.stringify(items, null, 2);
  document.getElementById("count").textContent = items.length + " events";
}

document.getElementById("refresh").onclick = render;

document.getElementById("export").onclick = async () => {
  const items = await getRecent(10000);
  const blob = new Blob([JSON.stringify(items)], { type: "application/json" });
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: "local-events.json" });
  a.click();
};

document.getElementById("exportTitles").onclick = async () => {
  const userId = (document.getElementById('userId').value || '12345').trim();
  const items = await getRecent(10000);
  // Titles come only from feed_video and shorts_video events now (page_title removed)
  const titles = [];
  const seen = new Set();
  for (const ev of items) {
    const t = (ev && ev.title && String(ev.title).trim()) || '';
    if (!t) continue;
    if (seen.has(t)) continue;
    seen.add(t);
    titles.push(t);
  }
  const out = { user_id: userId, titles };
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' });
  const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: 'titles.json' });
  a.click();
};

document.getElementById("clear").onclick = async () => { await clearAll(); render(); };

render();
