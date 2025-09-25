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

document.getElementById("clear").onclick = async () => { await clearAll(); render(); };

render();
