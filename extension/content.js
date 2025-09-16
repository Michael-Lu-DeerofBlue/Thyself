import { addEvent } from "./db.js";
(function () {
  const url = location.href;
  const title = document.title;
  let visibleSince = document.visibilityState === "visible" ? Date.now() : null;
  let dwellMs = 0, clicks = 0;

  function flush(reason) {
    addEvent({ ts: Date.now(), url, title, dwellMs, clicks, reason }).catch(()=>{});
  }
  document.addEventListener("click", () => { clicks += 1; }, { capture: true });

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden" && visibleSince) {
      dwellMs += Date.now() - visibleSince; visibleSince = null; flush("hidden");
    } else if (document.visibilityState === "visible") {
      visibleSince = Date.now();
    }
  });

  window.addEventListener("beforeunload", () => {
    if (visibleSince) dwellMs += Date.now() - visibleSince;
    flush("unload");
  });

  function hookVideos() {
    document.querySelectorAll("video").forEach(v => {
      const meta = () => ({ url, title, src: v.currentSrc, dur: v.duration, cur: v.currentTime });
      if (!v._loggerBound) {
        v._loggerBound = true;
        v.addEventListener("play",  () => addEvent({ ts: Date.now(), type: "video_play",  ...meta() }));
        v.addEventListener("pause", () => addEvent({ ts: Date.now(), type: "video_pause", ...meta() }));
        v.addEventListener("ended", () => addEvent({ ts: Date.now(), type: "video_ended", ...meta() }));
      }
    });
  }
  hookVideos();
  new MutationObserver(hookVideos).observe(document.documentElement, { childList: true, subtree: true });
})();
