// Content script focused solely on harvesting YouTube feed & shorts titles.
(function() {
  function send(payload) { try { chrome.runtime.sendMessage({ type: 'LCL_ADD_EVENT', payload }); } catch(_) {} }

  // -------------------- Robust YouTube feed harvesting --------------------
  if (location.hostname.includes('youtube.com')) {
    const seen = new Set(); // key = videoId|page
    let positionCounter = 0;
    let currentPath = location.pathname;

    function extractVideoIdFromHref(href) {
      if (!href) return null;
      const u = href.startsWith('http') ? href : ('https://www.youtube.com' + href);
      const m = u.match(/[?&]v=([A-Za-z0-9_-]{6,})/);
      return m ? m[1] : null;
    }

    // Heuristic extraction for desktop nodes
    function extractDesktop(node) {
      if (node.querySelector('ytd-display-ad-renderer,ytd-ad-slot-renderer, ytd-reel-shelf-renderer')) return null; // skip ads & shorts shelf
      // The video renderer might be nested inside rich item
      let anchor = node.querySelector('a#video-title-link, a#thumbnail[href^="/watch"], a#video-title[href^="/watch"], h3 a[href^="/watch"]');
      if (!anchor) return null;
      // Title strategies
      let titleEl = node.querySelector('#video-title, a#video-title, yt-formatted-string#video-title');
      if (!titleEl && anchor.getAttribute('title')) titleEl = anchor;
      let rawTitle = titleEl ? (titleEl.textContent || titleEl.getAttribute('title') || '').trim() : '';
      if (!rawTitle) {
        // Fallback: aria-label often contains title + metadata
        const aria = anchor.getAttribute('aria-label');
        if (aria) {
          const parts = aria.split(' by ');
          rawTitle = (parts && parts.length) ? parts[0] : aria;
        }
      }
      rawTitle = rawTitle.replace(/\s+/g, ' ').trim();
      if (!rawTitle) return null;
      const channelEl = node.querySelector('#channel-name a, ytd-channel-name a, a.yt-simple-endpoint.yt-formatted-string');
      const channel = channelEl ? channelEl.textContent.trim() : '';
      const lengthEl = node.querySelector('ytd-thumbnail-overlay-time-status-renderer span');
      const length = lengthEl ? lengthEl.textContent.trim() : '';
      const badges = Array.from(node.querySelectorAll('ytd-badge-supported-renderer')).map(b => b.textContent.trim()).filter(Boolean);
      return { anchor, rawTitle, channel, length, badges };
    }

    // Mobile extraction
    function extractMobile(node) {
      if (node.querySelector('ad-slot-render, ytm-promoted-sparkles-text-search-renderer')) return null;
      const anchor = node.querySelector('a[href^="/watch"]');
      if (!anchor) return null;
      let titleEl = anchor.querySelector('span.yt-core-attributed-string[role="text"], h3, .media-item-headline');
      let rawTitle = titleEl ? titleEl.textContent.trim() : '';
      if (!rawTitle && anchor.getAttribute('aria-label')) rawTitle = anchor.getAttribute('aria-label').trim();
      rawTitle = rawTitle.replace(/\s+/g, ' ').trim();
      if (!rawTitle) return null;
      const channelEl = node.querySelector('.media-item-metadata a, a.yt-core-attributed-string');
      const channel = channelEl ? channelEl.textContent.trim() : '';
      const lengthEl = node.querySelector('span.yt-core-attributed-string.ytd-thumbnail-overlay-time-status-renderer');
      const length = lengthEl ? lengthEl.textContent.trim() : '';
      return { anchor, rawTitle, channel, length, badges: [] };
    }

    // Nodes provider: returns unified list of candidate host nodes
    function candidateNodes() {
      // Desktop rich items
      const desktop = Array.from(document.querySelectorAll('ytd-rich-item-renderer'));
      if (desktop.length) return { variant: 'desktop', nodes: desktop };
      const mobile = Array.from(document.querySelectorAll('ytm-rich-item-renderer, ytm-video-with-context-renderer'));
      return { variant: 'mobile', nodes: mobile };
    }

    // Track nodes that need re-check because title not yet hydrated
    const pendingNodes = new WeakSet();

    function processNode(node, variant) {
      try {
        const extracted = variant === 'desktop' ? extractDesktop(node) : extractMobile(node);
        if (!extracted) {
          // Maybe not hydrated yet; schedule retry if it has a watch link
          const maybeLink = node.querySelector('a[href^="/watch"]');
          if (maybeLink && !pendingNodes.has(node)) {
            pendingNodes.add(node);
            setTimeout(() => processNode(node, variant), 1200);
          }
          return;
        }
        let { anchor, rawTitle, channel, length, badges } = extracted;
        const href = anchor.getAttribute('href') || '';
        const videoId = extractVideoIdFromHref(href);
        if (!videoId) return;
        const key = videoId + '|' + currentPath;
        if (seen.has(key)) return;
        // --- Clean trailing duration words from title & populate length if absent ---
        const durationSuffixRe = /(?:\s*-?\s*)?(\d+\s+hours?(?:,\s*\d+\s+minutes?)?(?:,\s*\d+\s+seconds?)?|\d+\s+minutes?(?:,\s*\d+\s+seconds?)?|\d+\s+seconds?)$/i;
        const m = rawTitle.match(durationSuffixRe);
        if (m) {
          const phrase = m[1];
          rawTitle = rawTitle.slice(0, m.index).trim();
          if (!length) {
            // Parse to a normalized hh:mm:ss or mm:ss if possible
            const parts = phrase.split(/,\s*/);
            let h=0, mi=0, s=0;
            parts.forEach(p => {
              const numMatch = p.match(/(\d+)\s+(hours?|hour|minutes?|minute|seconds?|second)/i);
              if (numMatch) {
                const val = parseInt(numMatch[1], 10);
                if (/hour/i.test(numMatch[2])) h = val; else if (/minute/i.test(numMatch[2])) mi = val; else if (/second/i.test(numMatch[2])) s = val;
              }
            });
            const pad = x => String(x).padStart(2,'0');
            if (h) length = `${h}:${pad(mi)}:${pad(s)}`; else if (mi || s) length = `${mi}:${pad(s)}`; else if (s) length = `0:${pad(s)}`;
          }
        }
        seen.add(key);
        positionCounter += 1;
        send({
          ts: Date.now(),
            type: 'feed_video',
            platform: 'youtube',
            variant,
            videoId,
            title: rawTitle,
            channel,
            length,
            badges,
            position: positionCounter,
            href: href.startsWith('http') ? href : 'https://www.youtube.com' + href,
            page: currentPath,
        });
      } catch (_) {}
    }

    // IntersectionObserver to process only visible (reduces noise & ensures hydrated)
    const io = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const variant = entry.target.closest('ytd-rich-item-renderer') ? 'desktop' : 'mobile';
          processNode(entry.target, variant);
        }
      });
    }, { rootMargin: '0px 0px 600px 0px', threshold: 0 });

    function scanAll() {
      const { variant, nodes } = candidateNodes();
      let considered = 0;
      nodes.forEach(n => {
        considered += 1;
        io.observe(n);
      });
    }

    // -------------------- Shorts harvesting (desktop & mobile) --------------------
    const shortsSeen = new Set(); // videoId|page
    function extractShortId(href) {
      if (!href) return null;
      const m = href.match(/\/shorts\/([A-Za-z0-9_-]{6,})/);
      return m ? m[1] : null;
    }
    function scanShorts() {
      try {
        const anchors = document.querySelectorAll('a[href^="/shorts/"]');
        let added = 0;
        anchors.forEach(a => {
          const href = a.getAttribute('href');
          const vid = extractShortId(href);
            if (!vid) return;
          const key = vid + '|' + currentPath;
          if (seen.has(key) || shortsSeen.has(key)) return;
          // Title precedence: inner span text, then title attribute, then textContent
          let tEl = a.querySelector('span[role="text"], span.yt-core-attributed-string');
          let rawTitle = tEl ? tEl.textContent.trim() : (a.getAttribute('title') || a.textContent || '').trim();
          rawTitle = rawTitle.replace(/\s+/g,' ').trim();
          if (!rawTitle) return;
          shortsSeen.add(key); seen.add(key);
          added += 1; positionCounter += 1;
          send({
            ts: Date.now(),
            type: 'shorts_video',
            platform: 'youtube',
            variant: /ytm-/.test(a.outerHTML) ? 'shorts_mobile' : 'shorts_desktop',
            videoId: vid,
            title: rawTitle,
            channel: '',
            length: '',
            badges: ['shorts'],
            position: positionCounter,
            href: href.startsWith('http') ? href : 'https://www.youtube.com' + href,
            page: currentPath,
          });
        });
      } catch (_) {}
    }

    // Mutation observer to catch dynamic appends
    const mo = new MutationObserver(muts => {
      for (const m of muts) {
        m.addedNodes && m.addedNodes.forEach(nd => {
          if (!(nd instanceof HTMLElement)) return;
          if (nd.matches && (nd.matches('ytd-rich-item-renderer') || nd.matches('ytm-rich-item-renderer') || nd.matches('ytm-video-with-context-renderer'))) {
            io.observe(nd);
          }
          if (nd.querySelector && nd.querySelector('a[href^="/shorts/"]')) {
            setTimeout(() => scanShorts(), 400);
          }
        });
      }
    });
    mo.observe(document.documentElement, { childList: true, subtree: true });

    // Periodic sweep (in case some nodes missed)
  setInterval(scanAll, 8000);
  setInterval(scanShorts, 9000);
  setTimeout(scanAll, 1000);
  setTimeout(scanShorts, 1400);

    // Navigation detection (YouTube SPA)
    function handleNavigation() {
      if (currentPath === location.pathname) return;
      currentPath = location.pathname;
      positionCounter = 0;
      seen.clear();
      setTimeout(scanAll, 700); // wait a bit for new feed load
      setTimeout(scanShorts, 900);
    }
    ['pushState','replaceState'].forEach(fn => {
      const orig = history[fn];
      history[fn] = function() { const r = orig.apply(this, arguments); setTimeout(handleNavigation, 0); return r; };
    });
    window.addEventListener('popstate', handleNavigation);
    document.addEventListener('yt-navigate-finish', () => setTimeout(handleNavigation, 0), true);
    document.addEventListener('yt-page-data-fetched', () => setTimeout(handleNavigation, 0), true);
  }
})();
