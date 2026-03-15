/* ============================================================
   app.js — Main application controller
   ADAI Training Metrics Dashboard — Tizen TV
   ============================================================
   Polls the ADAI metrics REST API and updates the dashboard.
   API docs: /docs/TRAINING_METRICS_API.md
   ============================================================ */

(function(window) {
    'use strict';

    /* -------------------------------------------------------
       Config & State
    ------------------------------------------------------- */
    var Config = {
        host:         localStorage.getItem('adai_host')     || '10.0.0.141',
        port:         localStorage.getItem('adai_port')     || '8081',
        pollInterval: parseInt(localStorage.getItem('adai_poll') || '2000', 10),
        chartMode:    localStorage.getItem('adai_chart')    || 'epochs',
    };

    var State = {
        connected:       false,
        training:        false,
        settingsOpen:    false,
        pollTimer:       null,
        retryCount:      0,
        maxRetry:        10,
        currentMetrics:  null,
        epochLosses:     [],
        epochValLosses:  [],
    };

    /* -------------------------------------------------------
       DOM references
    ------------------------------------------------------- */
    var $ = function(id) { return document.getElementById(id); };

    var UI = {
        /* Header */
        sessionBadge:       $('session-badge'),
        connectionStatus:   $('connection-status'),
        connectionLabel:    $('connection-label'),
        clock:              $('clock'),

        /* Primary metrics */
        epochValue:         $('epoch-value'),
        epochSub:           $('epoch-sub'),
        epochProgressBar:   $('epoch-progress-bar'),
        sampleValue:        $('sample-value'),
        sampleSub:          $('sample-sub'),
        sampleProgressBar:  $('sample-progress-bar'),
        etaValue:           $('eta-value'),
        etaSub:             $('eta-sub'),
        throughputValue:    $('throughput-value'),

        /* Loss metrics */
        lossValue:          $('loss-value'),
        runningLossValue:   $('running-loss-value'),
        bestEpochValue:     $('best-epoch-value'),
        valLossValue:       $('val-loss-value'),
        bestValLossValue:   $('best-val-loss-value'),
        runningValLossValue:$('running-val-loss-value'),

        /* Aux metrics */
        lrValue:            $('lr-value'),
        perplexityValue:    $('perplexity-value'),
        gradNormValue:      $('grad-norm-value'),

        /* Validation perplexity & accuracy (TD-015) */
        valPerplexityValue: $('val-perplexity-value'),
        valAccuracyValue:   $('val-accuracy-value'),

        /* Advanced diagnostics (TD-013) */
        gradVarianceValue:  $('gradient-variance-value'),
        computeRatioValue:  $('compute-ratio-value'),
        weightUpdateValue:  $('weight-update-value'),

        /* Session stats */
        totalSamplesValue:  $('total-samples-value'),
        sessionIdValue:     $('session-id-value'),
        totalTimeValue:     $('total-time-value'),
        bestEpochStatValue: $('best-epoch-stat-value'),

        /* Footer */
        lastUpdate:         $('last-update'),
        pollIntervalLabel:  $('poll-interval-label'),

        /* Overlays */
        settingsOverlay:    $('settings-overlay'),
        loadingOverlay:     $('loading-overlay'),
        loadingUrl:         $('loading-url'),

        /* Settings inputs */
        apiHostInput:       $('api-host-input'),
        apiPortInput:       $('api-port-input'),
        settingsSaveBtn:    $('settings-save-btn'),
        settingsCancelBtn:  $('settings-cancel-btn'),
        apiUrlPreview:      $('api-url-preview'),
    };

    /* -------------------------------------------------------
       Utilities
    ------------------------------------------------------- */
    function apiBase() {
        return 'http://' + Config.host + ':' + Config.port;
    }

    function fmt(val, decimals) {
        if (val === null || val === undefined || isNaN(val)) return '—';
        if (typeof val === 'number') return val.toFixed(decimals !== undefined ? decimals : 4);
        return String(val);
    }

    function fmtInt(val) {
        if (val === null || val === undefined || isNaN(val)) return '—';
        return Math.round(val).toLocaleString();
    }

    function fmtTime(seconds) {
        if (seconds === null || seconds === undefined || isNaN(seconds) || seconds < 0) return '—';
        seconds = Math.round(seconds);
        var h = Math.floor(seconds / 3600);
        var m = Math.floor((seconds % 3600) / 60);
        var s = seconds % 60;
        if (h > 0) return h + 'h ' + pad2(m) + 'm ' + pad2(s) + 's';
        if (m > 0) return pad2(m) + 'm ' + pad2(s) + 's';
        return s + 's';
    }

    function pad2(n) { return n < 10 ? '0' + n : String(n); }

    /* -------------------------------------------------------
       Dial Gauge helpers
    ------------------------------------------------------- */
    function gaugeArcPath(cx, cy, r, fromDeg, toDeg) {
        /* SVG sweep-flag=1 (CW in screen space) curves UPWARD through top of dial */
        var x1  = cx + r * Math.cos(fromDeg * Math.PI / 180);
        var y1  = cy - r * Math.sin(fromDeg * Math.PI / 180);
        var x2  = cx + r * Math.cos(toDeg   * Math.PI / 180);
        var y2  = cy - r * Math.sin(toDeg   * Math.PI / 180);
        var big = (fromDeg - toDeg > 180) ? 1 : 0;
        return 'M ' + x1.toFixed(2) + ' ' + y1.toFixed(2) +
               ' A ' + r + ' ' + r + ' 0 ' + big + ' 1 ' +
               x2.toFixed(2) + ' ' + y2.toFixed(2);
    }

    /* Build static gauge SVG into container div.
       ticks: [{val, label}] positioned within [min, max]. */
    function initGauge(containerId, min, max, warnThresh, errThresh, ticks) {
        var el = document.getElementById(containerId);
        if (!el) return;
        var cx = 100, cy = 92, R = 70;
        var wDeg = 180 - ((warnThresh - min) / (max - min)) * 180;
        var eDeg = 180 - ((errThresh  - min) / (max - min)) * 180;

        var tickSvg = '';
        if (ticks) {
            for (var i = 0; i < ticks.length; i++) {
                var pct = (ticks[i].val - min) / (max - min);
                var deg = 180 - pct * 180;
                var rad = deg * Math.PI / 180;
                var lx1 = cx + (R - 8)  * Math.cos(rad);
                var ly1 = cy - (R - 8)  * Math.sin(rad);
                var lx2 = cx + (R + 8)  * Math.cos(rad);
                var ly2 = cy - (R + 8)  * Math.sin(rad);
                var tx  = cx + (R + 20) * Math.cos(rad);
                var ty  = cy - (R + 20) * Math.sin(rad);
                tickSvg += '<line class="gauge-tick"' +
                    ' x1="' + lx1.toFixed(1) + '" y1="' + ly1.toFixed(1) + '"' +
                    ' x2="' + lx2.toFixed(1) + '" y2="' + ly2.toFixed(1) + '"/>';
                tickSvg += '<text x="' + tx.toFixed(1) + '" y="' + (ty + 5).toFixed(1) +
                    '" text-anchor="middle" class="gauge-tick-label">' + ticks[i].label + '</text>';
            }
        }

        el.innerHTML =
            '<svg viewBox="0 0 200 108" xmlns="http://www.w3.org/2000/svg">' +
            '<path class="gauge-band-green" d="' + gaugeArcPath(cx, cy, R, 180, wDeg) + '"/>' +
            '<path class="gauge-band-amber" d="' + gaugeArcPath(cx, cy, R, wDeg, eDeg)+ '"/>' +
            '<path class="gauge-band-red"   d="' + gaugeArcPath(cx, cy, R, eDeg, 0)   + '"/>' +
            '<g>' + tickSvg + '</g>' +
            '<line class="gauge-needle" id="' + containerId + '-needle"' +
              ' x1="' + cx + '" y1="' + cy + '" x2="' + (cx - R + 8) + '" y2="' + cy + '"/>' +
            '<circle class="gauge-center" cx="' + cx + '" cy="' + cy + '" r="6"/>' +
            '</svg>';
    }

    /* Rotate needle to current value */
    var gaugeState = {}; /* {containerId: {currentRad, animId}} */

    /* Ease-in-out (smoothstep) lerp */
    function easeInOut(t) {
        return t * t * (3 - 2 * t);
    }

    function setNeedleRad(needle, rad, cx, cy, R) {
        needle.setAttribute('x2', (cx + R * Math.cos(rad)).toFixed(2));
        needle.setAttribute('y2', (cy - R * Math.sin(rad)).toFixed(2));
    }

    function updateGauge(containerId, value, min, max) {
        var needle = document.getElementById(containerId + '-needle');
        if (!needle) return;
        var cx = 100, cy = 92, R = 63;
        var pct    = (value == null || isNaN(value)) ? 0
                   : Math.max(0, Math.min(1, (value - min) / (max - min)));
        var target = (1 - pct) * Math.PI; /* 180°→0° in radians */

        var gs = gaugeState[containerId];
        if (!gs) {
            gs = { currentRad: Math.PI, animId: null };
            gaugeState[containerId] = gs;
        }

        /* Cancel any running animation */
        if (gs.animId) {
            cancelAnimationFrame(gs.animId);
            gs.animId = null;
        }

        var startRad = gs.currentRad;
        var delta    = target - startRad;
        if (Math.abs(delta) < 0.001) return; /* nothing to animate */

        var duration = Math.min(800, Math.max(300, Math.abs(delta) / Math.PI * 900));
        var startTime = null;

        function step(ts) {
            if (!startTime) startTime = ts;
            var elapsed = ts - startTime;
            var t = Math.min(elapsed / duration, 1);
            var rad = startRad + delta * easeInOut(t);
            gs.currentRad = rad;
            setNeedleRad(needle, rad, cx, cy, R);
            if (t < 1) {
                gs.animId = requestAnimationFrame(step);
            } else {
                gs.animId = null;
            }
        }

        gs.animId = requestAnimationFrame(step);
    }

    /* -------------------------------------------------------
       Range-colour helpers
       applyRangeColor: green < warnThresh, amber < errThresh, else red
       applyLRColor:    green 1e-5..1e-3, amber nearby, else red
    ------------------------------------------------------- */
    function applyRangeColor(el, val, warnThresh, errThresh) {
        el.classList.remove('val-good', 'val-warn', 'val-error');
        if (val == null || isNaN(val)) return;
        if (val < warnThresh)     el.classList.add('val-good');
        else if (val < errThresh) el.classList.add('val-warn');
        else                      el.classList.add('val-error');
    }

    function applyLRColor(el, lr) {
        el.classList.remove('val-good', 'val-warn', 'val-error');
        if (lr == null || isNaN(lr)) return;
        if (lr >= 1e-5 && lr <= 1e-3)     el.classList.add('val-good');
        else if (lr > 0  && lr <= 1e-2)   el.classList.add('val-warn');
        else                              el.classList.add('val-error');
    }

    function fmtLR(lr) {
        if (!lr && lr !== 0) return '—';
        /* Scientific notation for very small values */
        if (lr < 0.0001) return lr.toExponential(3);
        return lr.toFixed(7).replace(/0+$/, '').replace(/\.$/, '');
    }

    function nowString() {
        var d = new Date();
        return d.toLocaleTimeString();
    }

    function flash(el) {
        if (!el) return;
        el.classList.remove('flash');
        /* Force reflow */
        void el.offsetWidth;
        el.classList.add('flash');
    }

    /* -------------------------------------------------------
       Clock
    ------------------------------------------------------- */
    function updateClock() {
        var d = new Date();
        UI.clock.textContent = pad2(d.getHours()) + ':' + pad2(d.getMinutes()) + ':' + pad2(d.getSeconds());
    }

    setInterval(updateClock, 1000);
    updateClock();

    /* -------------------------------------------------------
       Connection status
    ------------------------------------------------------- */
    function setConnected(status, label) {
        UI.connectionStatus.className = 'status-dot ' + status;
        UI.connectionLabel.textContent = label;
    }

    /* -------------------------------------------------------
       Fetch helpers
    ------------------------------------------------------- */
    function fetchJSON(endpoint, timeout) {
        timeout = timeout || 5000;
        var url = apiBase() + endpoint;

        return new Promise(function(resolve, reject) {
            var abortTimer = setTimeout(function() {
                reject(new Error('Timeout: ' + url));
            }, timeout);

            fetch(url, { mode: 'cors', cache: 'no-cache' })
                .then(function(r) {
                    clearTimeout(abortTimer);
                    if (!r.ok) throw new Error('HTTP ' + r.status);
                    return r.json();
                })
                .then(resolve)
                .catch(function(err) {
                    clearTimeout(abortTimer);
                    reject(err);
                });
        });
    }

    /* -------------------------------------------------------
       Dashboard update
    ------------------------------------------------------- */
    function applyMetrics(current, epochData) {
        /* --- Epoch progress --- */
        var epoch     = current.current_epoch      || 0;
        var totEpoch  = current.total_epochs       || 0;
        var epochPct  = totEpoch > 0 ? (epoch / totEpoch) * 100 : 0;

        setText(UI.epochValue, totEpoch > 0 ? epoch + ' / ' + totEpoch : fmt(epoch, 0));
        setText(UI.epochSub,   totEpoch > 0 ? Math.round(epochPct) + '%  complete' : 'Epoch ' + epoch);
        UI.epochProgressBar.style.width = Math.min(epochPct, 100) + '%';

        /* --- Sample progress --- */
        var sample    = current.current_sample  || 0;
        var totSample = current.total_samples   || 0;
        var samplePct = totSample > 0 ? (sample / totSample) * 100 : 0;

        setText(UI.sampleValue, totSample > 0 ? fmtInt(sample) + ' / ' + fmtInt(totSample) : fmtInt(sample));
        setText(UI.sampleSub,   (totSample > 0 ? Math.round(samplePct) + '% —' : '') + ' total: ' + fmtInt(current.total_samples_trained));
        UI.sampleProgressBar.style.width = Math.min(samplePct, 100) + '%';

        /* --- ETA & time --- */
        setText(UI.etaValue, fmtTime(current.estimated_time_remaining_seconds));
        setText(UI.etaSub,   'Elapsed: ' + fmtTime(current.total_training_time_seconds));

        /* --- Throughput --- */
        var sps = current.samples_per_second;
        setText(UI.throughputValue, sps ? fmt(sps, 3) : '—');
        adaptPollInterval(sps);

        /* --- Loss --- */
        setText(UI.lossValue,          fmt(current.current_loss));
        setText(UI.runningLossValue,   fmt(current.running_loss));
        setText(UI.bestEpochValue,     fmt(current.best_epoch, 0));
        applyRangeColor(UI.lossValue, current.current_loss, 2.0, 4.0);
        updateGauge('gauge-loss', current.current_loss, 0, 6);

        /* --- Validation Loss --- */
        setText(UI.valLossValue,           fmt(current.current_validation_loss));
        setText(UI.bestValLossValue,       fmt(current.best_validation_loss));
        setText(UI.runningValLossValue,    fmt(current.running_validation_loss));
        applyRangeColor(UI.valLossValue, current.current_validation_loss, 2.5, 5.0);

        /* --- Aux --- */
        setText(UI.lrValue,            fmtLR(current.current_learning_rate));
        setText(UI.perplexityValue,    fmt(current.current_perplexity, 4));
        setText(UI.gradNormValue,      fmt(current.current_gradient_norm, 4));
        applyLRColor(UI.lrValue,          current.current_learning_rate);
        applyRangeColor(UI.perplexityValue, current.current_perplexity,    50,  500);
        applyRangeColor(UI.gradNormValue,   current.current_gradient_norm, 1.0, 10.0);
        updateGauge('gauge-perplexity', current.current_perplexity,    0, 750);
        updateGauge('gauge-grad-norm',  current.current_gradient_norm, 0,  20);

        /* --- Validation Perplexity & Accuracy (TD-015) --- */
        var valPerplx = current.current_validation_perplexity;
        setText(UI.valPerplexityValue, (valPerplx && valPerplx > 0) ? fmt(valPerplx, 2) : '—');
        var valAcc = current.current_validation_accuracy;
        setText(UI.valAccuracyValue, (valAcc !== undefined && valAcc !== null && valAcc >= 0) ? fmt(valAcc * 100, 1) + '%' : '—');

        /* --- Advanced Diagnostics (TD-013) --- */
        setText(UI.gradVarianceValue, fmt(current.gradient_variance, 6));
        var ctr = current.compute_time_ratio;
        setText(UI.computeRatioValue, (ctr != null && !isNaN(ctr)) ? fmt(ctr * 100, 1) + '%' : '—');
        setText(UI.weightUpdateValue, fmt(current.weight_update_ratio, 6));

        /* --- Session stats --- */
        setText(UI.totalSamplesValue,  fmtInt(current.total_samples_trained));
        setText(UI.sessionIdValue,     fmt(current.session_id, 0));
        setText(UI.totalTimeValue,     fmtTime(current.total_training_time_seconds));
        setText(UI.bestEpochStatValue, fmt(current.best_epoch, 0));

        /* --- Session badge --- */
        if (current.is_training) {
            UI.sessionBadge.textContent  = 'SESSION ' + (current.session_id || '—') + '  ACTIVE';
            UI.sessionBadge.className    = 'badge badge-active';
        } else {
            UI.sessionBadge.textContent = 'SESSION ' + (current.session_id || '—') + '  IDLE';
            UI.sessionBadge.className   = 'badge badge-done';
        }

        /* --- Chart --- */
        if (epochData && chart) {
            var tl = epochData.epoch_losses          || [];
            var vl = epochData.epoch_validation_losses || [];
            chart.update(tl, vl);
        }

        /* Footer */
        UI.lastUpdate.textContent = 'Last update: ' + nowString();
    }

    function setText(el, val) {
        if (!el) return;
        var s = (val === null || val === undefined) ? '—' : String(val);
        if (el.textContent !== s) {
            el.textContent = s;
            flash(el);
        }
    }

    /* -------------------------------------------------------
       Polling
    ------------------------------------------------------- */
    function poll() {
        var p1 = fetchJSON('/api/metrics/current');
        var p2 = fetchJSON('/api/session/epochs');

        Promise.all([p1, p2])
            .then(function(results) {
                State.retryCount = 0;
                State.connected  = true;
                setConnected('connected', 'Connected');

                if (UI.loadingOverlay) {
                    UI.loadingOverlay.classList.add('hidden');
                }

                applyMetrics(results[0], results[1]);
            })
            .catch(function(err) {
                State.connected = false;
                State.retryCount++;

                var label = 'Reconnecting (' + State.retryCount + ')';
                setConnected('connecting', label);

                if (State.retryCount >= State.maxRetry) {
                    setConnected('disconnected', 'Disconnected');
                }

                console.warn('[ADAI] Poll error:', err.message);
            });
    }

    /* Adapt poll interval to match data production rate (1000 / sps ms).
       Clamped to [200 ms, 5000 ms]. Only restarts timer when the new interval
       differs from the current one by more than 10 %. */
    function adaptPollInterval(sps) {
        if (!sps || sps <= 0 || isNaN(sps)) return;
        var newInterval = Math.max(200, Math.min(5000, Math.round(1000 / sps)));
        if (Math.abs(newInterval - Config.pollInterval) / Config.pollInterval > 0.10) {
            Config.pollInterval = newInterval;
            stopPolling();
            State.pollTimer = setInterval(poll, Config.pollInterval);
            updatePollLabel();
        }
    }

    function updatePollLabel() {
        var sps = Config.pollInterval > 0 ? (1000 / Config.pollInterval).toFixed(2) : '—';
        UI.pollIntervalLabel.textContent =
            'Polling: ' + (Config.pollInterval / 1000).toFixed(2) + 's  (' + sps + ' polls/s)';
    }

    function startPolling() {
        stopPolling();
        poll(); /* immediate first poll */
        State.pollTimer = setInterval(poll, Config.pollInterval);
        updatePollLabel();
    }

    function stopPolling() {
        if (State.pollTimer) {
            clearInterval(State.pollTimer);
            State.pollTimer = null;
        }
    }

    /* -------------------------------------------------------
       Settings overlay
    ------------------------------------------------------- */
    function openSettings() {
        State.settingsOpen = true;
        UI.settingsOverlay.classList.remove('hidden');

        /* Populate current values */
        UI.apiHostInput.value = Config.host;
        UI.apiPortInput.value = Config.port;
        updateUrlPreview();

        /* Highlight current interval */
        document.querySelectorAll('.settings-btn[data-interval]').forEach(function(btn) {
            var iv = parseInt(btn.getAttribute('data-interval'), 10);
            btn.classList.toggle('selected', iv === Config.pollInterval);
        });

        /* Highlight current chart mode */
        document.querySelectorAll('.settings-btn[data-chart]').forEach(function(btn) {
            btn.classList.toggle('selected', btn.getAttribute('data-chart') === Config.chartMode);
        });

        nav.refresh();
        if (UI.apiHostInput) UI.apiHostInput.focus();
    }

    function closeSettings() {
        State.settingsOpen = false;
        UI.settingsOverlay.classList.add('hidden');
        nav.refresh();
        nav.focusById('card-settings');
    }

    function saveSettings() {
        Config.host         = UI.apiHostInput.value.trim() || '10.0.0.141';
        Config.port         = UI.apiPortInput.value.trim() || '8081';

        /* Read selected interval */
        var selInterval = document.querySelector('.settings-btn.selected[data-interval]');
        if (selInterval) Config.pollInterval = parseInt(selInterval.getAttribute('data-interval'), 10);

        /* Read selected chart mode */
        var selChart = document.querySelector('.settings-btn.selected[data-chart]');
        if (selChart) Config.chartMode = selChart.getAttribute('data-chart');

        localStorage.setItem('adai_host',  Config.host);
        localStorage.setItem('adai_port',  Config.port);
        localStorage.setItem('adai_poll',  String(Config.pollInterval));
        localStorage.setItem('adai_chart', Config.chartMode);

        closeSettings();

        /* Reconnect */
        State.retryCount = 0;
        setConnected('connecting', 'Connecting…');
        UI.loadingOverlay.classList.remove('hidden');
        UI.loadingUrl.textContent = apiBase();

        startPolling();
    }

    function updateUrlPreview() {
        var h = UI.apiHostInput.value.trim() || '…';
        var p = UI.apiPortInput.value.trim() || '8081';
        UI.apiUrlPreview.textContent = 'http://' + h + ':' + p;
    }

    /* -------------------------------------------------------
       Chart
    ------------------------------------------------------- */
    var chart = null;

    function initChart() {
        try {
            chart = new LossChart('loss-chart');
        } catch (e) {
            console.warn('[ADAI] Chart init failed:', e);
        }
    }

    /* -------------------------------------------------------
       Navigation wiring
    ------------------------------------------------------- */
    var nav = new TVNav();

    function initNavigation() {
        nav.refresh();

        /* OK on settings card → open settings */
        nav.on('ok', function(el) {
            if (!el) return;
            if (el.id === 'card-settings') {
                openSettings();
                return;
            }
            if (el.id === 'settings-save-btn')   { saveSettings();   return; }
            if (el.id === 'settings-cancel-btn')  { closeSettings();  return; }

            /* Interval buttons */
            var interval = el.getAttribute('data-interval');
            if (interval) {
                document.querySelectorAll('.settings-btn[data-interval]').forEach(function(b) {
                    b.classList.remove('selected');
                });
                el.classList.add('selected');
                updateUrlPreview();
            }

            /* Chart mode buttons */
            var chartMode = el.getAttribute('data-chart');
            if (chartMode) {
                document.querySelectorAll('.settings-btn[data-chart]').forEach(function(b) {
                    b.classList.remove('selected');
                });
                el.classList.add('selected');
            }
        });

        nav.on('back', function() {
            if (State.settingsOpen) closeSettings();
        });

        nav.on('blue', function() {
            openSettings();
        });

        nav.on('green', function() {
            /* Manual refresh */
            poll();
        });
    }

    /* Update URL preview live as user types (host/port fields) */
    function initSettingsInputs() {
        [UI.apiHostInput, UI.apiPortInput].forEach(function(input) {
            if (!input) return;
            input.addEventListener('input', updateUrlPreview);
            input.addEventListener('focus', function() { input.classList.add('focused'); });
            input.addEventListener('blur',  function() { input.classList.remove('focused'); });
        });

        UI.settingsSaveBtn.addEventListener('click', saveSettings);
        UI.settingsCancelBtn.addEventListener('click', closeSettings);
    }

    /* -------------------------------------------------------
       Init
    ------------------------------------------------------- */
    function init() {
        /* Show loading */
        UI.loadingUrl.textContent = apiBase();

        /* Init dial gauges */
        initGauge('gauge-loss', 0, 6, 2.0, 4.0,
            [{val:0,label:'0'},{val:2,label:'2'},{val:4,label:'4'},{val:6,label:'6'}]);
        initGauge('gauge-perplexity', 0, 750, 50, 500,
            [{val:0,label:'0'},{val:50,label:'50'},{val:500,label:'500'},{val:750,label:'750'}]);
        initGauge('gauge-grad-norm', 0, 20, 1.0, 10.0,
            [{val:0,label:'0'},{val:1,label:'1'},{val:10,label:'10'},{val:20,label:'20'}]);

        /* Init chart */
        initChart();

        /* Wire settings inputs */
        initSettingsInputs();

        /* Init navigation */
        initNavigation();

        /* Footer */
        updatePollLabel();

        /* Register Tizen key events (required on Samsung TV) */
        try {
            tizen.tvinputdevice.registerKeyBatch([
                'ColorF0Red', 'ColorF1Green', 'ColorF2Yellow', 'ColorF3Blue',
                'Info', 'MediaPlay', 'MediaPause'
            ]);
        } catch (e) {
            /* Not running on Tizen — browser preview mode */
        }

        /* Start polling */
        startPolling();
    }

    document.addEventListener('DOMContentLoaded', init);

})(window);
