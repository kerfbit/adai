/* ============================================================
   navigation.js — TV D-pad / remote-control navigation
   ADAI Training Metrics Dashboard
   ============================================================
   Samsung TV remote key codes (Tizen):
     Up    = 38   Down  = 40   Left  = 37   Right = 39
     OK    = 13   Back  = 10009
     Exit  = 10182
     Color keys: Red=403, Green=404, Yellow=405, Blue=406
   ============================================================ */

(function(window) {
    'use strict';

    var KEY = {
        UP:     38,
        DOWN:   40,
        LEFT:   37,
        RIGHT:  39,
        OK:     13,
        BACK:   10009,
        EXIT:   10182,
        RED:    403,
        GREEN:  404,
        YELLOW: 405,
        BLUE:   406,
        PLAY:   415,
        PAUSE:  19,
        INFO:   457,
    };

    /**
     * TVNav — manages focus movement through a flat list of focusable elements.
     * Provides spatial navigation (up/down/left/right) and an event bus.
     */
    function TVNav() {
        this._focusables = [];          /* all currently focusable elements */
        this._focused    = null;        /* currently focused element */
        this._listeners  = {};          /* event listeners */

        this._onKeyDown = this._onKeyDown.bind(this);
        document.addEventListener('keydown', this._onKeyDown);
    }

    /** Register a named event listener. */
    TVNav.prototype.on = function(event, cb) {
        if (!this._listeners[event]) this._listeners[event] = [];
        this._listeners[event].push(cb);
    };

    TVNav.prototype._emit = function(event, data) {
        var list = this._listeners[event];
        if (list) list.forEach(function(cb) { cb(data); });
    };

    /** Rebuild the list of focusable elements from the DOM. */
    TVNav.prototype.refresh = function() {
        this._focusables = Array.prototype.slice.call(
            document.querySelectorAll('.focusable')
        ).filter(function(el) {
            return el.offsetParent !== null; /* exclude hidden elements */
        });

        /* If nothing focused yet, focus first */
        if (!this._focused || !document.contains(this._focused)) {
            this._focusFirst();
        }
    };

    TVNav.prototype._focusFirst = function() {
        if (this._focusables.length > 0) {
            this._setFocus(this._focusables[0]);
        }
    };

    TVNav.prototype._setFocus = function(el) {
        if (!el) return;
        if (this._focused) {
            this._focused.classList.remove('focused');
        }
        this._focused = el;
        el.classList.add('focused');
        el.focus();
        this._emit('focus', el);
    };

    TVNav.prototype.focusById = function(id) {
        var el = document.getElementById(id);
        if (el) this._setFocus(el);
    };

    /**
     * Get the geometric center of a DOM element.
     */
    function center(el) {
        var r = el.getBoundingClientRect();
        return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
    }

    /**
     * From the current element, find the best neighbor in `direction`.
     * Uses a distance-weighted nearest-neighbor search.
     *
     * @param {Element} from
     * @param {'up'|'down'|'left'|'right'} direction
     * @returns {Element|null}
     */
    TVNav.prototype._findNeighbor = function(from, direction) {
        var focusables = this._focusables;
        var fc = center(from);

        var best     = null;
        var bestScore = Infinity;

        focusables.forEach(function(el) {
            if (el === from) return;
            var ec = center(el);
            var dx = ec.x - fc.x;
            var dy = ec.y - fc.y;

            var inDirection = false;
            switch (direction) {
                case 'up':    inDirection = dy < -10; break;
                case 'down':  inDirection = dy >  10; break;
                case 'left':  inDirection = dx < -10; break;
                case 'right': inDirection = dx >  10; break;
            }
            if (!inDirection) return;

            /* Prefer elements along the primary axis */
            var primary, secondary;
            if (direction === 'up' || direction === 'down') {
                primary   = Math.abs(dy);
                secondary = Math.abs(dx);
            } else {
                primary   = Math.abs(dx);
                secondary = Math.abs(dy);
            }

            /* Weight: penalize off-axis distance */
            var score = primary + secondary * 2.5;
            if (score < bestScore) {
                bestScore = score;
                best = el;
            }
        });

        return best;
    };

    /** Handle all keyboard/remote events. */
    TVNav.prototype._onKeyDown = function(e) {
        switch (e.keyCode) {
            case KEY.UP:
                e.preventDefault();
                this._move('up');
                break;
            case KEY.DOWN:
                e.preventDefault();
                this._move('down');
                break;
            case KEY.LEFT:
                e.preventDefault();
                this._move('left');
                break;
            case KEY.RIGHT:
                e.preventDefault();
                this._move('right');
                break;
            case KEY.OK:
                e.preventDefault();
                if (this._focused) {
                    this._emit('ok', this._focused);
                    this._focused.click();
                }
                break;
            case KEY.BACK:
                e.preventDefault();
                this._emit('back', this._focused);
                break;
            case KEY.EXIT:
                e.preventDefault();
                this._emit('exit', null);
                try { tizen.application.getCurrentApplication().exit(); } catch(ex) { /* no-op in browser */ }
                break;
            case KEY.BLUE:
                e.preventDefault();
                this._emit('blue', null);
                break;
            case KEY.GREEN:
                e.preventDefault();
                this._emit('green', null);
                break;
            case KEY.RED:
                e.preventDefault();
                this._emit('red', null);
                break;
            case KEY.YELLOW:
                e.preventDefault();
                this._emit('yellow', null);
                break;
            case KEY.INFO:
                e.preventDefault();
                this._emit('info', null);
                break;
            default:
                break;
        }
    };

    TVNav.prototype._move = function(direction) {
        if (!this._focused) {
            this._focusFirst();
            return;
        }
        var neighbor = this._findNeighbor(this._focused, direction);
        if (neighbor) {
            this._setFocus(neighbor);
        }
    };

    TVNav.prototype.destroy = function() {
        document.removeEventListener('keydown', this._onKeyDown);
    };

    /* Export */
    window.TVNav = TVNav;
    window.TV_KEY = KEY;

})(window);
