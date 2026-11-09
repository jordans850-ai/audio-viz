/* ── Audio Visualizer AI — Frontend ────────────────────────────── */

// ─── Token handling ───────────────────────────────────────────────
// Read token from URL query param or cookie, and include in all API calls.
(function() {
  const params = new URLSearchParams(window.location.search);
  const urlToken = params.get('token') || null;
  const cookies = document.cookie.split(';');
  let cookieToken = null;
  for (const c of cookies) {
    const [name, val] = c.trim().split('=');
    if (name === 'viz_token') cookieToken = val;
  }
  window.AUTH_TOKEN = urlToken || cookieToken;

  // If token came from URL, persist it in cookie and clean URL
  if (urlToken) {
    const d = new Date();
    d.setTime(d.getTime() + 7 * 24 * 60 * 60 * 1000);
    document.cookie = `viz_token=${urlToken};expires=${d.toUTCString()};path=/;samesite=lax`;
    const url = new URL(window.location);
    url.searchParams.delete('token');
    window.history.replaceState({}, '', url);
  }
})();

function apiFetch(path, options = {}) {
  const url = '/api' + path;
  const token = window.AUTH_TOKEN;
  const headers = Object.assign({}, options.headers);
  let finalUrl = url;

  if (token) {
    if (!options.method || options.method === 'GET') {
      const sep = url.includes('?') ? '&' : '?';
      finalUrl = url + sep + 'token=' + encodeURIComponent(token);
    } else {
      headers['X-Auth-Token'] = token;
    }
  }
  return fetch(finalUrl, Object.assign({}, options, { headers }));
}
  { id: 'cosmic-space',       name: 'Cosmic Space',   desc: 'Stars, nebulae, deep space glow' },
  { id: 'neon-cyberpunk',    name: 'Neon Cyberpunk', desc: 'Electric neon lines, dark city' },
  { id: 'ocean-deep',        name: 'Ocean Deep',      desc: 'Deep blue waves, bioluminescence' },
  { id: 'firestorm',         name: 'Firestorm',       desc: 'Fire, embers, intense heat' },
  { id: 'geometric-minimal', name: 'Geometric',       desc: 'Clean shapes, minimal palette' },
  { id: 'vaporwave',         name: 'Vaporwave',       desc: 'Retro synth aesthetic, pastels' },
  { id: 'nature-forest',     name: 'Forest',          desc: 'Green leaves, organic motion' },
  { id: 'dark-gothic',       name: 'Dark Gothic',     desc: 'Dark crystals, purple smoke' },
  { id: 'retro-80s',         name: 'Retro 80s',       desc: 'Stars, synthwave glow' },
  { id: 'abstract-particles',name: 'Particles',       desc: 'Chaotic glowing particles' },
  { id: 'anime-style',       name: 'Anime Style',     desc: 'Sparkles, rings, Japanese aesthetic' },
  { id: 'photo-realistic',   name: 'Light Rays',      desc: 'Cinematic light beams' },
];

const AI_ICON = '🤖'; // AI keyframe indicator

const form       = document.getElementById('vizForm');
const submitBtn  = document.getElementById('submitBtn');
const statusSec  = document.getElementById('statusSection');
const statusId   = document.getElementById('statusId');
const statusBadge= document.getElementById('statusBadge');
const statusMsg  = document.getElementById('statusMessage');
const progressFill = document.getElementById('progressFill');
const progressPct  = document.getElementById('progressPercent');
const progressSub  = document.getElementById('progressSub');
const statusActions= document.getElementById('statusActions');
const downloadBtn  = document.getElementById('downloadBtn');
const newBtn       = document.getElementById('newBtn');
const themeModal   = document.getElementById('themeModal');
const themeGrid    = document.getElementById('themeGrid');
const modalClose   = document.getElementById('modalClose');
const paletteBtn   = document.getElementById('themePaletteBtn');
const sourceSelect = document.getElementById('sourceSelect');

let currentJobId = null;
let pollInterval = null;

/* ── Theme modal ────────────────────────────────────────────────── */
function renderThemeGrid() {
  themeGrid.innerHTML = THEMES.map(t => {
    const colors = themeColors(t.id);
    const swatch = colors.map(c =>
      `<span style="flex:1;background:${c}"></span>`
    ).join('');
    return `
      <div class="theme-card" data-id="${t.id}">
        <div class="theme-swatch">${swatch}</div>
        <div class="theme-name">${t.name}</div>
        <div class="theme-desc">${t.desc}</div>
      </div>`;
  }).join('');
  themeGrid.querySelectorAll('.theme-card').forEach(card => {
    card.addEventListener('click', () => {
      const id = card.dataset.id;
      document.getElementById('theme').value = id;
      themeModal.classList.add('hidden');
    });
  });
}

function themeColors(id) {
  const palettes = {
    'cosmic-space':     ['#050514','#3a5ac8','#8855ff','#64c8ff'],
    'neon-cyberpunk':   ['#05001a','#00ffc8','#ff0064','#ff64ff'],
    'ocean-deep':       ['#000a20','#003c99','#00bac8','#64ffc8'],
    'firestorm':        ['#0f0200','#ff3a00','#ffc832','#ff9632'],
    'geometric-minimal':['#14142a','#ececff','#b4b4ff','#d8d8ff'],
    'vaporwave':        ['#14002a','#ff64c8','#64c8ff','#ff96dc'],
    'nature-forest':    ['#051405','#1ea82e','#64ffa0','#80ff80'],
    'dark-gothic':      ['#00000d','#500050','#b40050','#960096'],
    'retro-80s':        ['#0a0018','#ff9632','#64c8ff','#ffc864'],
    'abstract-particles':['#0a0515','#ff6464','#64c8ff','#ff96c8'],
    'anime-style':      ['#14143a','#ff78c8','#64c8ff','#ffc8dc'],
    'photo-realistic':  ['#141420','#ececff','#d4d4d4','#ffffff'],
  };
  return palettes[id] || ['#14142a','#a78bfa','#7c5cfc','#c9c2e6'];
}

paletteBtn.addEventListener('click', () => {
  renderThemeGrid();
  themeModal.classList.remove('hidden');
});
modalClose.addEventListener('click', () => themeModal.classList.add('hidden'));
themeModal.querySelector('.modal-backdrop').addEventListener('click', () =>
  themeModal.classList.add('hidden'));

/* ── Form submit ────────────────────────────────────────────────── */
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (submitBtn.disabled) return;

  const url      = document.getElementById('url').value.trim();
  const theme    = document.getElementById('theme').value.trim() || 'default';
  const resolution = document.getElementById('resolution').value;
  const fps      = parseInt(document.getElementById('fps').value);

  // Spotify validation
  if (sourceSelect.value === 'spotify' && !url.includes('spotify.com')) {
    statusMsg.textContent = 'That does not look like a Spotify link.';
    return;
  }

  submitBtn.disabled = true;
  submitBtn.innerHTML = '<span class="btn-icon">⟳</span> Processing…';

  const fd = new FormData(form);
  fd.append('url', url);
  fd.append('theme', theme);
  fd.append('resolution', resolution);
  fd.append('fps', fps);

  try {
    const res = await apiFetch('/api/process', { method: 'POST', body: fd });
    const job = await res.json();
    currentJobId = job.id || Object.keys(jobs || {}).pop();
    // job id returned from post; store it
    currentJobId = (await res.json()).id;
    startPolling(currentJobId);
  } catch (err) {
    statusMsg.textContent = 'Request failed: ' + err.message;
    submitBtn.disabled = false;
    submitBtn.innerHTML = '<span class="btn-icon">✦</span> Generate Visuals';
  }
});

/* ── Status display ─────────────────────────────────────────────── */
function showStatus(id, job) {
  statusSec.classList.remove('hidden');
  statusId.textContent = 'Job #' + id;
  statusMsg.innerHTML = job.message.replace(
    /(1920|1080|1440|4K|BPM|beats|frames|seconds|MB|KB|AI|keyframe|Keyframe)/g,
    '<span class="highlight">$1</span>'
  );

  // Show AI keyframe info if available
  if (job.ai_prompts && job.ai_prompts.length > 0 && job.status !== 'done') {
    const promptSection = document.getElementById('aiPromptSection');
    if (promptSection) {
      promptSection.style.display = 'block';
      renderAIPrompts(job.ai_prompts);
    }
  }

  const badgeMap = {
    queued: '',
    downloading: 'downloading',
    analyzing: '',
    generating: 'generating',
    encoding: 'encoding',
    done: 'done',
    error: 'error',
  };
  statusBadge.textContent = job.status;
  statusBadge.className = 'status-badge ' + (badgeMap[job.status] || '');

  progressFill.style.width = (job.progress || 0) + '%';
  progressPct.textContent = Math.round(job.progress || 0) + '%';
  progressSub.textContent = job.message.split('.')[0] || '';

  if (job.status === 'done') {
    downloadBtn.href = '/api/download/' + id;
    statusActions.style.display = 'flex';
    submitBtn.disabled = false;
    submitBtn.innerHTML = '<span class="btn-icon">✦</span> Generate Visuals';
  } else if (job.status === 'error') {
    statusActions.style.display = 'flex';
    submitBtn.disabled = false;
    submitBtn.innerHTML = '<span class="btn-icon">✦</span> Generate Visuals';
  } else {
    statusActions.style.display = 'none';
  }
}

function renderAIPrompts(prompts) {
  const container = document.getElementById('aiPromptList');
  if (!container) return;
  container.innerHTML = prompts.map((p, i) => {
    const hasImage = p.image_path ? '✓' : '○';
    const label = p.image_path ? 'Visual ready' : 'Prompt generated — generate image';
    return `
      <div class="ai-prompt-item">
        <div class="ai-prompt-header">
          <span class="ai-prompt-time">t=${p.time.toFixed(1)}s</span>
          <span class="ai-prompt-intensity">${p.intensity.toFixed(2)} intensity</span>
          <span class="ai-prompt-status">${hasImage} ${label}</span>
        </div>
        <div class="ai-prompt-text">${p.prompt}</div>
      </div>`;
  }).join('');
}

function startPolling(id) {
  stopPolling();
  pollInterval = setInterval(async () => {
    try {
      const res = await apiFetch('/api/status/' + id);
      const job = await res.json();
      showStatus(id, job);
      // Fetch AI prompts separately (not in status response)
      if (job.status === 'generating' || job.status === 'done') {
        try {
          const pr = await apiFetch('/api/keyframes/prompts/' + id);
          const prompts = await pr.json();
          if (prompts.prompts && prompts.prompts.length > 0) {
            job.ai_prompts = prompts.prompts;
            showStatus(id, job);
          }
        } catch {}
      }
      if (job.status === 'done' || job.status === 'error') {
        stopPolling();
      }
    } catch {
      stopPolling();
    }
  }, 800);
}

function stopPolling() {
  if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
}

newBtn.addEventListener('click', () => {
  statusSec.classList.add('hidden');
  currentJobId = null;
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

/* ── Source select affects URL placeholder ─────────────────────── */
sourceSelect.addEventListener('change', () => {
  const placeholders = {
    youtube:    'https://youtube.com/watch?v=...',
    soundcloud: 'https://soundcloud.com/artist/track',
    spotify:    'https://open.spotify.com/track/...',
  };
  document.getElementById('url').placeholder = placeholders[sourceSelect.value] || '';
});

/* ── Demo beat animation in header ─────────────────────────────── */
(function demoBeat() {
  const logoAccent = document.querySelector('.logo .accent');
  if (!logoAccent) return;
  let beat = 0;
  setInterval(() => {
    const t = Date.now() / 1000;
    // simulate a 128 BPM pulse
    const phase = (t * 128 / 60) % 1;
    const glow = 0.6 + 0.4 * Math.sin(phase * Math.PI * 2);
    logoAccent.style.textShadow = `0 0 ${12 * glow}px rgba(167,139,250,${0.3 * glow})`;
  }, 50);
})();

/* ── Fix: fetch returns id in response ─────────────────────────── */
// Patch: the /api/process endpoint returns the job directly; we need to
// extract job_id from it. Since jobs dict keys are UUID-based, we use a
// small workaround — ask status endpoint to find the latest job.
async function getLatestJobId() {
  // Simple approach: enumerate known status endpoints isn't available,
  // so we rely on the response containing the job_id directly.
  // The backend returns the ProcessingStatus; we augment it.
  return null;
}
