document.addEventListener('DOMContentLoaded', function () {
// ------------------------------------------------------------------
    // Moves every Bootstrap modal to <body> so it escapes the .main-content
    // stacking context (position: relative; z-index: 1). Modals nested inside
    // that context are painted BELOW Bootstrap's .modal-backdrop (z-index
    // 1040, appended at <body> root), leaving an invisible layer that swallows
    // every click over the dialog — frozen close (x) button and unresponsive
    // native video controls. Reparenting restores stacking on every page.
    // ------------------------------------------------------------------
    document.querySelectorAll('.modal').forEach(function (modalEl) {
        if (modalEl.parentElement && modalEl.parentElement !== document.body) {
            document.body.appendChild(modalEl);
        }
    });
// ------------------------------------------------------------------
    // Modal helpers - open/close any Bootstrap modal. When the Bootstrap JS
    // bundle is available its API is used; otherwise the container is
    // force-shown/hidden directly so the close (x) button ALWAYS works even
    // if the CDN script is blocked or fails to load.
    // ------------------------------------------------------------------
    function closeModal(modalEl) {
        if (!modalEl) return;
        // 1) ALWAYS force-hide the container directly so the pop-up can NEVER
        //    get stuck open, even if Bootstrap's instance is out of sync (e.g.
        //    the modal was opened without a live Bootstrap instance, or its
        //    hide transition never completed).
        modalEl.classList.remove('show');
        modalEl.style.display = 'none';
        modalEl.setAttribute('aria-hidden', 'true');
        document.body.classList.remove('modal-open');
        document.querySelectorAll('.modal-backdrop').forEach(function (b) { b.remove(); });
        if (modalEl._fallbackBackdrop) {
            modalEl._fallbackBackdrop = null;
        }

        // 2) Best-effort: tell Bootstrap the modal is closing so its own
        //    cleanup events (hidden.bs.modal, focus restore) still fire.
        if (window.bootstrap && window.bootstrap.Modal) {
            try {
                const instance = bootstrap.Modal.getInstance(modalEl);
                if (instance) { instance.hide(); }
            } catch (err) { /* already force-hidden above */ }
        }

        // 3) Stop any media still playing behind the closed pop-up.
        const vid = modalEl.querySelector('video');
        if (vid) { vid.pause(); vid.removeAttribute('src'); vid.load(); }
    }

    function openModal(modalEl) {
        if (!modalEl) return;
        // Prefer Bootstrap's managed show (backdrop, focus, aria)...
        if (window.bootstrap && window.bootstrap.Modal) {
            try {
                const instance = bootstrap.Modal.getInstance(modalEl) || bootstrap.Modal.getOrCreateInstance(modalEl);
                instance.show();
                return;
            } catch (err) { /* fall through to forced show */ }
        }
        // ...but ALWAYS make the pop-up visible even if Bootstrap is missing.
        modalEl.style.display = 'block';
        modalEl.classList.add('show');
        modalEl.setAttribute('aria-hidden', 'false');
        document.body.classList.add('modal-open');
        // Dimming backdrop so clicking outside the pop-up closes it too.
        if (!modalEl._fallbackBackdrop) {
            document.querySelectorAll('.modal-backdrop').forEach(function (b) { b.remove(); });
            const backdrop = document.createElement('div');
            backdrop.className = 'modal-backdrop fade show';
            document.body.appendChild(backdrop);
            modalEl._fallbackBackdrop = backdrop;
            backdrop.addEventListener('click', function () { closeModal(modalEl); });
        }
    }

    // Expose helpers so page-level inline scripts can reuse them too.
    window.closeModal = closeModal;
    window.openModal = openModal;
// ------------------------------------------------------------------
    // Generic modal close handler: every × button inside any Bootstrap modal
    // (History preview/details, recording review, detection popups, etc.)
    // explicitly hides its `.modal` container. Guarantees popups always close
    // even when the data-bs-dismiss data API is unavailable or blocked.
    // ------------------------------------------------------------------
    document.addEventListener('click', function (event) {
        const target = event.target;
        const closeBtn = target && target.closest
            ? target.closest('.modal .btn-close, .modal [data-bs-dismiss="modal"]')
            : null;
        if (!closeBtn) return;
        const modalEl = closeBtn.closest('.modal');
        if (!modalEl) return;
        event.preventDefault();
        event.stopPropagation();
        closeModal(modalEl);
    });

    // ------------------------------------------------------------------
    // Detection History "Actions" dropdown uses Bootstrap's standard
    // data-bs-toggle="dropdown" handling: the menu is anchored strictly to
    // its parent toggle button via CSS (position: absolute; top: 100%;
    // right: 0) and is never repositioned based on cursor coordinates.
    // ------------------------------------------------------------------
    const sidebarToggle = document.getElementById('sidebarToggle');
    const sidebar = document.querySelector('.sidebar');
    const themeToggle = document.querySelector('.theme-toggle');
    const html = document.documentElement;
    // Settings dark-mode checkbox (may be present on settings page)
    const darkModeToggle = document.getElementById('darkModeToggle');

    if (sidebarToggle && sidebar) {
        sidebarToggle.addEventListener('click', function () {
            sidebar.classList.toggle('show');
        });
    }

    // Theme init and toggle with persistence
    const THEME_KEY = 'site-theme';
    const WALLPAPER_KEY = 'site-wallpaper';
    const LANGUAGE_KEY = 'site-language';
    // Settings: theme select element (may be present on settings page)
    const themeSelect = document.getElementById('themeSelect');

    function setTheme(theme) {
        html.setAttribute('data-theme', theme);
        if (themeToggle) {
            themeToggle.innerHTML = theme === 'dark' ? '<i class="bi bi-sun"></i>' : '<i class="bi bi-moon"></i>';
        }
        // apply wallpaper if present — ALL themes; the CSS overlay keeps it subtle
        try {
            const wp = localStorage.getItem(WALLPAPER_KEY);
            if (wp) {
                document.documentElement.style.setProperty('--wallpaper-url', `url('${wp}')`);
                document.documentElement.classList.add('has-wallpaper');
            } else {
                document.documentElement.style.removeProperty('--wallpaper-url');
                document.documentElement.classList.remove('has-wallpaper');
            }
        } catch (e) {
            // ignore storage errors
        }
    }

    // Load saved theme or default from html attribute
    let saved = null;
    try { saved = localStorage.getItem(THEME_KEY); } catch (e) { saved = null; }
    const initialTheme = saved || (html.getAttribute('data-theme') || 'light');
    setTheme(initialTheme);

    // keep the Settings dropdown in sync with the active theme
    if (themeSelect) {
        const hasOption = Array.prototype.some.call(
            themeSelect.options,
            function (o) { return o.value === initialTheme; }
        );
        if (hasOption) { themeSelect.value = initialTheme; }
    }

    if (themeToggle) {
        themeToggle.addEventListener('click', function () {
            const currentTheme = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
            setTheme(currentTheme);
            try { localStorage.setItem(THEME_KEY, currentTheme); } catch (e) { /* ignore */ }
            // sync settings toggle if present
            try { if (darkModeToggle) darkModeToggle.checked = (currentTheme === 'dark'); } catch (e) {}
        });
    }

    // Settings page controls (language, wallpaper)
    const languageSelect = document.getElementById('languageSelect');
    const wallpaperInput = document.getElementById('wallpaperInput');
    const uploadWallpaperBtn = document.getElementById('uploadWallpaperBtn');
    const removeWallpaperBtn = document.getElementById('removeWallpaperBtn');
    const wallpaperPreview = document.getElementById('wallpaperPreview');
    const saveSettingsBtn = document.getElementById('saveSettingsBtn');

    // initialize language select from storage

    // Basic client-side translation dictionaries for interface keys
    const translations = {
        "en": {
            "nav.dashboard": "Dashboard",
            "nav.image": "Image Detection",
            "nav.video": "Video Detection",
            "nav.webcam": "Webcam Detection",
            "nav.history": "Detection History",
            "nav.analytics": "Analytics",
            "nav.reports": "Reports",
            "nav.admin": "Admin Panel",
            "nav.settings": "Settings",
            "nav.profile": "Profile",
            "nav.logout": "Logout",
            "settings.title": "Settings",
            "settings.dark_mode": "Dark Mode",
            "settings.language": "Language",
            "settings.theme": "Theme",
            "settings.upload_wallpaper": "Upload Wallpaper / Watermark",
            "settings.remove_wallpaper": "Remove Wallpaper",
            "settings.upload": "Upload",
            "settings.notifications": "Notifications",
            "settings.wallpaper": "Custom Wallpaper / Watermark (Dark Mode)",
            "admin.manage_users": "Manage Users",
            "admin.admin_controls": "Admin Controls",
        },

        "hi": {
            "nav.dashboard": "डैशबोर्ड",
            "nav.image": "इमेज डिटेक्शन",
            "nav.video": "वीडियो डिटेक्शन",
            "nav.webcam": "वेबकैम डिटेक्शन",
            "nav.history": "डिटेक्शन इतिहास",
            "nav.analytics": "विश्लेषण",
            "nav.reports": "रिपोर्ट्स",
            "nav.admin": "एडमिन पैनल",
            "nav.settings": "सेटिंग्स",
            "nav.profile": "प्रोफ़ाइल",
            "nav.logout": "लॉग आउट",
            "settings.title": "सेटिंग्स",
            "settings.dark_mode": "डार्क मोड",
            "settings.language": "भाषा",
            "settings.theme": "थीम",
            "settings.upload_wallpaper": "वॉलपेपर/वॉटरमार्क अपलोड",
            "settings.remove_wallpaper": "वॉलपेपर हटाएं",
            "admin.manage_users": "उपयोगकर्ताओं का प्रबंधन",
            "admin.admin_controls": "एडमिन नियंत्रण",
        },
        "es": {
            "nav.dashboard": "Tablero",
            "nav.image": "Detección de Imágenes",
            "nav.video": "Detección de Video",
            "nav.webcam": "Detección Webcam",
            "nav.history": "Historial",
            "nav.analytics": "Analíticas",
            "nav.reports": "Informes",
            "nav.admin": "Panel Admin",
            "nav.settings": "Ajustes",
            "nav.profile": "Perfil",
            "nav.logout": "Cerrar Sesión",
            "settings.title": "Ajustes",
            "settings.dark_mode": "Modo Oscuro",
            "settings.language": "Idioma",
            "settings.theme": "Tema",
            "settings.upload_wallpaper": "Subir Fondo / Marca",
            "settings.remove_wallpaper": "Eliminar Fondo",
            "admin.manage_users": "Administrar Usuarios",
            "admin.admin_controls": "Controles Admin",
        },
        "fr": {
            "nav.dashboard": "Tableau",
            "nav.image": "Détection Image",
            "nav.video": "Détection Vidéo",
            "nav.webcam": "Détection Webcam",
            "nav.history": "Historique",
            "nav.analytics": "Analytique",
            "nav.reports": "Rapports",
            "nav.admin": "Panneau Admin",
            "nav.settings": "Paramètres",
            "nav.profile": "Profil",
            "nav.logout": "Déconnexion",
            "settings.title": "Paramètres",
            "settings.dark_mode": "Mode Sombre",
            "settings.language": "Langue",
            "settings.theme": "Thème",
            "settings.upload_wallpaper": "Téléverser Fond/Filigrane",
            "settings.remove_wallpaper": "Supprimer Fond",
            "admin.manage_users": "Gérer les Utilisateurs",
            "admin.admin_controls": "Contrôles Admin",
        },
        "de": {
            "nav.dashboard": "Dashboard",
            "nav.image": "Bild-Erkennung",
            "nav.video": "Video-Erkennung",
            "nav.webcam": "Webcam-Erkennung",
            "nav.history": "Verlauf",
            "nav.analytics": "Analyse",
            "nav.reports": "Berichte",
            "nav.admin": "Admin-Panel",
            "nav.settings": "Einstellungen",
            "nav.profile": "Profil",
            "nav.logout": "Abmelden",
            "settings.title": "Einstellungen",
            "settings.dark_mode": "Dunkler Modus",
            "settings.language": "Sprache",
            "settings.theme": "Thema",
            "settings.upload_wallpaper": "Hintergrund hochladen",
            "settings.remove_wallpaper": "Hintergrund entfernen",
            "admin.manage_users": "Benutzer verwalten",
            "admin.admin_controls": "Admin-Tools",
        },
        "zh": {
            "nav.dashboard": "仪表板",
            "nav.image": "图像检测",
            "nav.video": "视频检测",
            "nav.webcam": "网络摄像头检测",
            "nav.history": "检测历史",
            "nav.analytics": "分析",
            "nav.reports": "报告",
            "nav.admin": "管理员面板",
            "nav.settings": "设置",
            "nav.profile": "个人资料",
            "nav.logout": "登出",
            "settings.title": "设置",
            "settings.dark_mode": "深色模式",
            "settings.language": "语言",
            "settings.theme": "主题",
            "settings.upload_wallpaper": "上传壁纸/水印",
            "settings.remove_wallpaper": "移除壁纸",
            "admin.manage_users": "管理用户",
            "admin.admin_controls": "管理员控制",
        },
        "ar": {
            "nav.dashboard": "لوحة التحكم",
            "nav.image": "كشف الصور",
            "nav.video": "كشف الفيديو",
            "nav.webcam": "كشف الكاميرا",
            "nav.history": "سجل الكشف",
            "nav.analytics": "التحليلات",
            "nav.reports": "التقارير",
            "nav.admin": "لوحة المشرف",
            "nav.settings": "الإعدادات",
            "nav.profile": "الملف الشخصي",
            "nav.logout": "تسجيل الخروج",
            "settings.title": "الإعدادات",
            "settings.dark_mode": "الوضع الداكن",
            "settings.language": "اللغة",
            "settings.theme": "المظهر",
            "settings.upload_wallpaper": "رفع الخلفية/العلامة",
            "settings.remove_wallpaper": "إزالة الخلفية",
            "admin.manage_users": "إدارة المستخدمين",
            "admin.admin_controls": "أدوات المشرف",
        },
        "ja": {},
        "ru": {},
        "pt": {},
        "it": {},
        "ko": {}
    };

    function applyTranslations(lang) {
        const dict = translations[lang] || translations['en'];
        document.querySelectorAll('[data-i18n]').forEach(function (el) {
            const key = el.getAttribute('data-i18n');
            if (dict[key]) el.textContent = dict[key];
        });
    }

    // initialize language selection and apply translations

    try {
        const lang = localStorage.getItem(LANGUAGE_KEY) || 'en';
        if (languageSelect && lang) {
            languageSelect.value = lang;
        }
        applyTranslations(lang);
    } catch (e) { }

    if (languageSelect) {
        languageSelect.addEventListener('change', function () {
            try { localStorage.setItem(LANGUAGE_KEY, this.value); } catch (e) {}
            applyTranslations(this.value);
        });
    }

    // theme select (Settings) handler
    if (themeSelect) {
        try { themeSelect.value = (html.getAttribute('data-theme') === 'dark') ? 'dark' : 'light'; } catch (e) {}
        themeSelect.addEventListener('change', function () {
            const val = this.value;
            const allowedThemes = ['light', 'dark', 'blue'];
            const newTheme = allowedThemes.includes(val) ? val : 'light';
            setTheme(newTheme);
            try { localStorage.setItem(THEME_KEY, newTheme); } catch (e) {}
            if (darkModeToggle) darkModeToggle.checked = (newTheme === 'dark');
            if (themeToggle) themeToggle.innerHTML = newTheme === 'dark' ? '<i class="bi bi-sun"></i>' : '<i class="bi bi-moon"></i>';
        });
    }

    // Secret-unlock: wallpaper section hidden by default; unlock via Advanced link or double-clicking the Settings title
    const WALLPAPER_SECRET = 'sahil123'; // configurable default in code
    const wallpaperSection = document.getElementById('wallpaperSection');
    const settingsTitleEl = document.getElementById('settingsTitle');
    const advancedLink = document.getElementById('advancedSettingsLink');

    function revealWallpaperSection() {
        try {
            if (!wallpaperSection) return;
            wallpaperSection.style.display = 'block';
            // show preview if a wallpaper URL is present
            try {
                const wp2 = localStorage.getItem(WALLPAPER_KEY);
                if (wp2 && wallpaperPreview) {
                    wallpaperPreview.src = wp2;
                    wallpaperPreview.classList.remove('d-none');
                }
            } catch (e) {}
            try { sessionStorage.setItem('wallpaper_unlocked', 'true'); } catch (e) {}
        } catch (e) { console.error('revealWallpaperSection error', e); }
    }

    function tryUnlockViaPrompt() {
        try {
            const attempt = prompt('Enter password:');
            if (attempt && String(attempt).trim() === WALLPAPER_SECRET) {
                revealWallpaperSection();
                setTimeout(function () { alert('Advanced settings unlocked for this session'); }, 200);
                return true;
            }
            alert('Incorrect password');
            return false;
        } catch (e) {
            console.error('Unlock failed', e);
            return false;
        }
    }

    try {
        // If previously unlocked in this session, reveal immediately
        const unlocked = sessionStorage.getItem('wallpaper_unlocked') === 'true';
        if (unlocked && wallpaperSection) {
            wallpaperSection.style.display = 'block';
        }
    } catch (e) { }

    if (settingsTitleEl) {
        settingsTitleEl.addEventListener('dblclick', function (ev) {
            tryUnlockViaPrompt();
        });
    }

    if (advancedLink) {
        advancedLink.addEventListener('click', function (ev) {
            ev.preventDefault();
            tryUnlockViaPrompt();
        });
    }

    // Lock / hide the advanced wallpaper section again.
    // Clears the session flag, instantly hides the section, and flashes a
    // small confirmation on the discreet Advanced Settings link.
    const lockSettingsBtn = document.getElementById('lockSettingsBtn');
    if (lockSettingsBtn) {
        lockSettingsBtn.addEventListener('click', function () {
            try { sessionStorage.removeItem('wallpaper_unlocked'); } catch (e) {}
            if (wallpaperSection) { wallpaperSection.style.display = 'none'; }
            if (advancedLink) {
                const originalLabel = 'Advanced Settings';
                advancedLink.innerHTML = '<i class="bi bi-lock-fill me-1"></i>Advanced settings locked';
                setTimeout(function () {
                    advancedLink.textContent = originalLabel;
                }, 2500);
            }
        });
    }

    // initialize wallpaper preview if set
    try {
        const wp = localStorage.getItem(WALLPAPER_KEY);
        const unlocked = sessionStorage.getItem('wallpaper_unlocked') === 'true';
        if (wp && wallpaperPreview) {
            wallpaperPreview.src = wp;
            // only reveal preview if wallpaper section already unlocked for this session
            if (unlocked && wallpaperSection) {
                wallpaperPreview.classList.remove('d-none');
            }
        }
        // sync theme toggle checkbox
        if (darkModeToggle) {
            darkModeToggle.checked = (html.getAttribute('data-theme') === 'dark');
            darkModeToggle.addEventListener('change', function () {
                const newTheme = this.checked ? 'dark' : 'light';
                setTheme(newTheme);
                try { localStorage.setItem(THEME_KEY, newTheme); } catch (e) {}
                // also sync topbar button icon
                if (themeToggle) themeToggle.innerHTML = newTheme === 'dark' ? '<i class="bi bi-sun"></i>' : '<i class="bi bi-moon"></i>';
            });
        }
    } catch (e) {}
    if (uploadWallpaperBtn && wallpaperInput) {
        uploadWallpaperBtn.addEventListener('click', async function () {
            if (!wallpaperInput.files || !wallpaperInput.files[0]) { alert('Select an image first'); return; }
            const file = wallpaperInput.files[0];
            const fd = new FormData(); fd.append('wallpaper', file);
            uploadWallpaperBtn.disabled = true; uploadWallpaperBtn.textContent = 'Uploading...';
            try {
                const resp = await fetch('/settings/upload_wallpaper', { method: 'POST', body: fd });
                const j = await resp.json();
                if (resp.ok && j.url) {
                    try { localStorage.setItem(WALLPAPER_KEY, j.url); } catch (e) {}
                    if (wallpaperPreview) { wallpaperPreview.src = j.url; wallpaperPreview.classList.remove('d-none'); }
                    // apply immediately for every theme
                    document.documentElement.style.setProperty('--wallpaper-url', `url('${j.url}')`);
                    document.documentElement.classList.add('has-wallpaper');
                } else {
                    alert('Upload failed: ' + (j.message || 'server error'));
                }
            } catch (err) {
                alert('Upload failed: ' + err.message);
            } finally { uploadWallpaperBtn.disabled = false; uploadWallpaperBtn.textContent = 'Upload'; }
        });
    }

    if (removeWallpaperBtn) {
        removeWallpaperBtn.addEventListener('click', async function () {
            try {
                const wp = localStorage.getItem(WALLPAPER_KEY);
                // attempt server-side removal only if stored path is under static/wallpapers
                let payload = { url: wp };
                const resp = await fetch('/settings/remove_wallpaper', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
                // ignore server errors, just clear local
                try { localStorage.removeItem(WALLPAPER_KEY); } catch (e) {}
                document.documentElement.style.removeProperty('--wallpaper-url');
                document.documentElement.classList.remove('has-wallpaper');
                if (wallpaperPreview) { wallpaperPreview.src = ''; wallpaperPreview.classList.add('d-none'); }
            } catch (e) {
                console.error(e);
            }
        });
    }

    if (saveSettingsBtn) {
        saveSettingsBtn.addEventListener('click', function () {
            // currently settings saved locally (theme, language). Could extend to server-side.
            try { localStorage.setItem(THEME_KEY, html.getAttribute('data-theme') || 'light'); } catch (e) {}
            try { localStorage.setItem(LANGUAGE_KEY, languageSelect ? languageSelect.value : 'en'); } catch (e) {}
            alert('Settings saved');
        });
    }

    // Admin: delete user buttons (AJAX) - prevents deleting primary admin (id 1) or self
    document.querySelectorAll('.btn-delete-user').forEach(function (btn) {
        btn.addEventListener('click', async function () {
            const userId = this.getAttribute('data-user-id');
            if (!userId) return;
            if (!confirm('Are you sure you want to delete this user?')) return;
            try {
                const resp = await fetch(`/admin/delete_user/${userId}`, { method: 'POST', headers: { 'Content-Type': 'application/json' } });
                const j = await resp.json();
                if (resp.ok && j.success) {
                    // remove table row
                    const tr = this.closest('tr'); if (tr) tr.remove();
                    alert('User deleted');
                } else {
                    alert('Could not delete user: ' + (j.message || 'Server error'));
                }
            } catch (err) {
                console.error(err); alert('Delete failed: ' + err.message);
            }
        });
    });

    const imageInput = document.getElementById('imageInput');
    const imagePreview = document.getElementById('imagePreview');
    if (imageInput && imagePreview) {
        imageInput.addEventListener('change', function () {
            const file = this.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function (event) {
                    imagePreview.src = event.target.result;
                    imagePreview.classList.remove('d-none');
                };
                reader.readAsDataURL(file);
            }
        });
    }

    const detectImageBtn = document.getElementById('detectImageBtn');
    if (detectImageBtn && imageInput) {
        detectImageBtn.addEventListener('click', async function () {
            if (!imageInput.files || !imageInput.files[0]) {
                alert('Please select an image first');
                return;
            }

            const formData = new FormData();
            formData.append('file', imageInput.files[0]);

            detectImageBtn.disabled = true;
            detectImageBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing...';

            try {
                const response = await fetch('/api/detect-image', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.success) {
                    const statusValue = document.getElementById('statusValue');
                    const confidenceValue = document.getElementById('confidenceValue');
                    const severityValue = document.getElementById('severityValue');
                    const summaryValue = document.getElementById('summaryValue');
                    const boundingBoxValue = document.getElementById('boundingBoxValue');
                    const processingValue = document.getElementById('processingValue');
                    const originalImageResult = document.getElementById('originalImageResult');
                    const detectedImageResult = document.getElementById('detectedImageResult');
                    const screenshotResult = document.getElementById('screenshotResult');
                    const screenshotPlaceholder = document.getElementById('screenshotPlaceholder');

                    if (statusValue) {
                        const isAccident = data.accident_status === 'Accident';
                        statusValue.textContent = isAccident ? '✅ Accident' : '✅ Non-Accident';
                        statusValue.className = `fs-4 fw-bold ${isAccident ? 'text-danger' : 'text-success'}`;
                    }

                    if (confidenceValue) {
                        confidenceValue.textContent = `${(Number(data.accident_confidence || 0) * 100).toFixed(2)}%`;
                    }

                    if (severityValue) {
                        const severity = (data.severity || 'low').toLowerCase();
                        const colorMap = { high: 'text-danger', medium: 'text-warning', low: 'text-success' };
                        severityValue.textContent = severity.charAt(0).toUpperCase() + severity.slice(1);
                        severityValue.className = `fs-4 fw-bold ${colorMap[severity] || 'text-secondary'}`;
                    }

                    if (boundingBoxValue) {
                        boundingBoxValue.textContent = `${data.total_objects} objects`;
                    }

                    if (processingValue) {
                        processingValue.textContent = `${data.inference_time}s`;
                    }

                    if (summaryValue) {
                        const summaryItems = data.summary_display || Object.entries(data.summary || {}).map(([label, count]) => ({ label, count }));
                        summaryValue.innerHTML = summaryItems.map(item => `
                            <div class="col-sm-6 col-lg-4">
                                <div class="p-3 bg-white rounded-4 border h-100 d-flex justify-content-between align-items-center">
                                    <span class="fw-semibold text-capitalize">${item.label}</span>
                                    <span class="badge bg-primary rounded-pill">${item.count}</span>
                                </div>
                            </div>
                        `).join('');
                    }

                    if (originalImageResult) {
                        originalImageResult.src = data.original_image;
                        originalImageResult.classList.remove('d-none');
                    }

                    if (detectedImageResult) {
                        detectedImageResult.src = data.detected_image;
                        detectedImageResult.classList.remove('d-none');
                    }

                    // Severity screenshot: only visible when an accident was detected.
                    if (screenshotResult && screenshotPlaceholder) {
                        if (data.screenshot) {
                            screenshotResult.src = data.screenshot;
                            screenshotResult.classList.remove('d-none');
                            screenshotPlaceholder.classList.add('d-none');
                        } else {
                            screenshotResult.classList.add('d-none');
                            screenshotResult.removeAttribute('src');
                            screenshotPlaceholder.classList.remove('d-none');
                        }
                    }
                } else {
                    alert('Error: ' + (data.error || 'Detection failed'));
                }
            } catch (error) {
                alert('Error: ' + error.message);
                console.error('Detection error:', error);
            } finally {
                detectImageBtn.disabled = false;
                detectImageBtn.innerHTML = 'Detect';
            }
        });
    }

    const imagePreviewModal = document.getElementById('imagePreviewModal');
    const imagePreviewModalImg = document.getElementById('imagePreviewModalImg');
    const videoPreviewModalVideo = document.getElementById('videoPreviewModalVideo');
    const videoPreviewModalSource = document.getElementById('videoPreviewModalSource');
    const imagePreviewModalLabel = document.getElementById('imagePreviewModalLabel');
    if (imagePreviewModal && imagePreviewModalImg && imagePreviewModalLabel) {
        // The modal instance is resolved lazily by the window.openModal /
        // window.closeModal helpers so nothing breaks if Bootstrap is missing.

        // Explicit close handler for the preview modal's close button. The button also
        // carries data-bs-dismiss="modal", but ensure the modal always closes by
        // targeting the modal container via its Bootstrap instance.
        const imagePreviewCloseBtn = imagePreviewModal.querySelector('.btn-close');
        if (imagePreviewCloseBtn) {
            imagePreviewCloseBtn.addEventListener('click', function () {
                closeModal(imagePreviewModal);
            });
        }

        document.querySelectorAll('.js-view-image').forEach(function (button) {
            button.addEventListener('click', function (e) {
                e.preventDefault();
                e.stopPropagation();
                const mediaUrl = this.getAttribute('data-image');
                const title = this.getAttribute('data-title') || 'Preview';
                imagePreviewModalLabel.textContent = title;

                // Detect video files by extension so MP4/webcam recordings play
                // in the <video> tag instead of showing a broken <img> preview.
                const isVideo = /\.(mp4|webm|mov|avi|mkv|flv)(\?.*)?$/i.test(mediaUrl || '');

                if (isVideo && videoPreviewModalVideo) {
                    // Reveal the player BEFORE load and point the <video> element
                    // directly at the file endpoint so the browser's native media
                    // controller can stream it (more reliable than a <source> src).
                    videoPreviewModalVideo.classList.remove('d-none');
                    imagePreviewModalImg.classList.add('d-none');
                    imagePreviewModalImg.removeAttribute('src');
                    videoPreviewModalVideo.src = mediaUrl;
                    videoPreviewModalVideo.preload = 'auto';
                    videoPreviewModalVideo.load();
                    const playPromise = videoPreviewModalVideo.play();
                    if (playPromise && playPromise.catch) {
                        playPromise.catch(function () { /* autoplay may be blocked; user can press play */ });
                    }
                } else {
                    imagePreviewModalImg.src = mediaUrl;
                    imagePreviewModalImg.classList.remove('d-none');
                    if (videoPreviewModalVideo) {
                        videoPreviewModalVideo.pause();
                        videoPreviewModalVideo.removeAttribute('src');
                        videoPreviewModalVideo.load();
                        videoPreviewModalVideo.classList.add('d-none');
                    }
                }
                openModal(imagePreviewModal);
            });
        });

        // Pause + clear video playback when the modal is closed
        imagePreviewModal.addEventListener('hidden.bs.modal', function () {
            if (videoPreviewModalVideo) {
                videoPreviewModalVideo.pause();
                videoPreviewModalVideo.removeAttribute('src');
                videoPreviewModalVideo.load();
            }
        });
    }

    // Webcam Detection Controls
    const startCameraBtn = document.getElementById('startCameraBtn');
    const stopCameraBtn = document.getElementById('stopCameraBtn');
    const applySourceBtn = document.getElementById('applySourceBtn');
    const cameraSourceInput = document.getElementById('cameraSourceInput');
    const cameraPlaceholder = document.getElementById('cameraPlaceholder');
    const videoStream = document.getElementById('videoStream');
    const fpsValue = document.getElementById('fpsValue');
    const objectsValue = document.getElementById('objectsValue');
    const framesValue = document.getElementById('framesValue');
    const statusValue = document.getElementById('statusValue');
    const detectionSummary = document.getElementById('detectionSummary');

    if (startCameraBtn && stopCameraBtn && videoStream) {
        let streamActive = false;
        let statsInterval = null;

        // Default camera source from config (0 = primary webcam / DroidCam)
        if (cameraSourceInput) {
            cameraSourceInput.value = '0';
        }

        function getCameraSource() {
            if (!cameraSourceInput) return '0';
            const val = cameraSourceInput.value.trim();
            return val || '0';
        }

        function startCamera() {
            const source = getCameraSource();
            const feedUrl = `/webcam_feed?source=${encodeURIComponent(source)}`;
            videoStream.src = feedUrl;
            videoStream.style.display = 'block';
            if (cameraPlaceholder) cameraPlaceholder.style.display = 'none';
            streamActive = true;
            startCameraBtn.disabled = true;
            stopCameraBtn.disabled = false;
            if (statusValue) {
                statusValue.textContent = 'Streaming';
                statusValue.className = 'fw-bold text-success';
            }

            // Poll stats periodically
            statsInterval = setInterval(fetchStats, 2000);
        }

        async function stopCamera() {
            streamActive = false;
            if (statsInterval) {
                clearInterval(statsInterval);
                statsInterval = null;
            }
            videoStream.style.display = 'none';
            if (cameraPlaceholder) cameraPlaceholder.style.display = 'flex';
            startCameraBtn.disabled = false;
            stopCameraBtn.disabled = true;
            if (statusValue) {
                statusValue.textContent = 'Idle';
                statusValue.className = 'fw-bold';
            }

            try {
                const source = getCameraSource();
                const response = await fetch(`/stop_webcam?source=${encodeURIComponent(source)}`);
                const data = await response.json();
                if (data.success && data.summary) {
                    const summary = data.summary;
                    if (fpsValue) fpsValue.textContent = summary.average_fps ? summary.average_fps.toFixed(1) : '0';
                    if (objectsValue) objectsValue.textContent = summary.total_objects_detected || 0;
                    if (framesValue) framesValue.textContent = summary.total_frames || 0;
                    if (detectionSummary && summary.object_counts_by_class && Object.keys(summary.object_counts_by_class).length > 0) {
                        detectionSummary.innerHTML = Object.entries(summary.object_counts_by_class)
                            .map(([cls, count]) => `<span class="badge bg-primary me-2 mb-1">${cls}: ${count}</span>`)
                            .join('');
                    }
                    if (statusValue && summary.status) {
                        statusValue.textContent = summary.status;
                        statusValue.className = summary.status === 'Accident' ? 'fw-bold text-danger' : 'fw-bold text-success';
                    }

                    // Session Result panel: Accident status, confidence, severity, screenshot.
                    const accidentStatusEl = document.getElementById('webcamAccidentStatus');
                    const confidenceEl = document.getElementById('webcamConfidence');
                    const severityEl = document.getElementById('webcamSeverity');
                    const screenshotEl = document.getElementById('webcamScreenshot');
                    const screenshotPlaceholderEl = document.getElementById('webcamScreenshotPlaceholder');

                    const isAccident = summary.status === 'Accident';
                    if (accidentStatusEl) {
                        accidentStatusEl.textContent = isAccident ? '✅ Accident' : '✅ Non-Accident';
                        accidentStatusEl.className = `fs-5 fw-bold ${isAccident ? 'text-danger' : 'text-success'}`;
                    }
                    if (confidenceEl) {
                        confidenceEl.textContent = `${(Number(summary.accident_confidence || 0) * 100).toFixed(2)}%`;
                    }
                    if (severityEl) {
                        const severity = String(data.severity || 'low').toLowerCase();
                        const colorMap = { high: 'text-danger', medium: 'text-warning', low: 'text-success' };
                        severityEl.textContent = severity.charAt(0).toUpperCase() + severity.slice(1);
                        severityEl.className = `fs-5 fw-bold ${colorMap[severity] || 'text-secondary'}`;
                    }
                    const screenshotUrl = data.screenshot_url;
                    if (screenshotEl && screenshotPlaceholderEl) {
                        if (screenshotUrl) {
                            screenshotEl.src = screenshotUrl;
                            screenshotEl.classList.remove('d-none');
                            screenshotPlaceholderEl.classList.add('d-none');
                        } else {
                            screenshotEl.classList.add('d-none');
                            screenshotEl.removeAttribute('src');
                            screenshotPlaceholderEl.classList.remove('d-none');
                        }
                    }
                }
            } catch (error) {
                console.error('Error stopping camera:', error);
            }
        }

        async function fetchStats() {
            try {
                const response = await fetch('/webcam_stats');
                const data = await response.json();
                if (data.success) {
                    if (fpsValue) fpsValue.textContent = data.fps ? data.fps.toFixed(1) : '0';
                    if (objectsValue) objectsValue.textContent = data.total_objects || 0;
                    if (framesValue) framesValue.textContent = data.frame_count || 0;
                    if (detectionSummary && data.summary && Object.keys(data.summary).length > 0) {
                        detectionSummary.innerHTML = Object.entries(data.summary)
                            .map(([cls, count]) => `<span class="badge bg-primary me-2 mb-1">${cls}: ${count}</span>`)
                            .join('');
                    }
                }
            } catch (error) {
                // Stats endpoint may not be available yet - ignore
            }
        }

        startCameraBtn.addEventListener('click', startCamera);
        stopCameraBtn.addEventListener('click', stopCamera);
        if (applySourceBtn) {
            applySourceBtn.addEventListener('click', function () {
                if (streamActive) {
                    stopCamera().then(() => {
                        startCamera();
                    });
                }
            });
        }
    }

    const videoInput = document.getElementById('videoInput');
    const videoPreview = document.getElementById('videoPreview');
    if (videoInput && videoPreview) {
        videoInput.addEventListener('change', function () {
            const file = this.files[0];
            if (file) {
                videoPreview.src = URL.createObjectURL(file);
                videoPreview.classList.remove('d-none');
            }
        });
    }

    const progressBar = document.getElementById('videoProgressBar');
    const detectVideoBtn = document.getElementById('detectVideoBtn');
    if (detectVideoBtn && videoInput && progressBar) {
        detectVideoBtn.addEventListener('click', async function () {
            if (!videoInput.files || !videoInput.files[0]) {
                alert('Please select a video first');
                return;
            }

            const formData = new FormData();
            formData.append('file', videoInput.files[0]);

            detectVideoBtn.disabled = true;
            detectVideoBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing...';
            progressBar.style.width = '0%';
            progressBar.classList.add('progress-bar-striped', 'progress-bar-animated');

            try {
                const response = await fetch('/api/detect-video', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.success) {
                    progressBar.style.width = '100%';

                    const accidentIncidents = data.accident_incidents !== undefined
                        ? data.accident_incidents
                        : (data.summary && data.summary.accident) || 0;

                    const isAccident = data.status === 'Accident';
                    const severity = String(data.severity || 'low').toLowerCase();
                    const severityColor = { high: 'bg-danger', medium: 'bg-warning text-dark', low: 'bg-success' }[severity] || 'bg-secondary';
                    const statusColor = isAccident ? 'text-danger' : 'text-success';

                    document.getElementById('videoResultValue').innerHTML = `
                        <div class="mt-2">
                            <strong>Detection Statistics:</strong>
                            <ul class="list-unstyled mt-2">
                                <li><i class="bi ${isAccident ? 'bi-exclamation-triangle text-danger' : 'bi-check-circle text-success'}"></i> Accident Status: <strong class="${statusColor}">${data.status || 'Non-Accident'}</strong></li>
                                <li><i class="bi bi-check-circle text-success"></i> Confidence: ${((data.accident_confidence || 0) * 100).toFixed(2)}%</li>
                                <li><i class="bi bi-activity"></i> Severity: <span class="badge ${severityColor}">${severity.charAt(0).toUpperCase() + severity.slice(1)}</span></li>
                                <li><i class="bi bi-check-circle text-success"></i> Total Objects: ${data.total_objects}</li>
                                <li><i class="bi bi-check-circle text-success"></i> Total Frames: ${data.total_frames}</li>
                                <li><i class="bi bi-check-circle text-success"></i> Processed Frames: ${data.processed_frames}</li>
                                <li><i class="bi bi-check-circle text-success"></i> Average Confidence: ${(data.average_confidence * 100).toFixed(2)}%</li>
                                <li><i class="bi bi-check-circle text-success"></i> Processing Time: ${data.inference_time}s</li>
                            </ul>
                            <strong class="mt-3 d-block">Detection Summary:</strong>
                            <ul class="list-unstyled mt-2">
                                <li><i class="bi bi-exclamation-triangle text-danger"></i> <strong>Accident Incidents (Unique):</strong> ${accidentIncidents}</li>
                                ${Object.entries(data.summary).filter(([key]) => key !== 'accident').map(([key, count]) => `<li>${key}: ${count}</li>`).join('')}
                            </ul>
                            ${data.screenshot ? `
                            <div class="mt-3">
                                <strong class="d-block">Accident Screenshot:</strong>
                                <img src="${data.screenshot}" alt="Accident screenshot" class="img-fluid rounded-3 border" style="max-height: 320px;">
                            </div>` : ''}
                        </div>
                    `;

                    const resultDiv = document.querySelector('.card.border-0.bg-light');
                    if (!resultDiv.querySelector('.row.g-3')) {
                        resultDiv.innerHTML += `
                            <div class="row g-3 mt-3">
                                <div class="col-md-6">
                                    <h6>Original Video</h6>
                                    <video controls class="w-100 rounded-3" style="max-height: 300px;">
                                        <source src="${data.original_video}" type="video/mp4">
                                    </video>
                                </div>
                                <div class="col-md-6">
                                    <h6>Detected Video</h6>
                                    <video controls class="w-100 rounded-3" style="max-height: 300px;">
                                        <source src="${data.detected_video}" type="video/mp4">
                                    </video>
                                </div>
                            </div>
                        `;
                    } else {
                        const videos = resultDiv.querySelectorAll('.row.g-3 video');
                        if (videos.length >= 2) {
                            videos[0].querySelector('source').src = data.original_video;
                            videos[1].querySelector('source').src = data.detected_video;
                            videos[0].load();
                            videos[1].load();
                        }
                    }

                    const downloadBtn = document.getElementById('downloadResultBtn');
                    if (downloadBtn) {
                        downloadBtn.onclick = function () {
                            const link = document.createElement('a');
                            link.href = data.detected_video;
                            link.download = 'detected_video.mp4';
                            document.body.appendChild(link);
                            link.click();
                            document.body.removeChild(link);
                        };
                        downloadBtn.disabled = false;
                    }
                } else {
                    alert('Error: ' + (data.error || 'Detection failed'));
                    progressBar.style.width = '0%';
                }
            } catch (error) {
                alert('Error: ' + error.message);
                console.error('Video detection error:', error);
                progressBar.style.width = '0%';
            } finally {
                detectVideoBtn.disabled = false;
                detectVideoBtn.innerHTML = 'Detect';
            }
        });
    }

});

