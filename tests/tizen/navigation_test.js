'use strict';

// Tests for tizen-metrics-app/js/navigation.js (TD-049) — TVNav's remote-control key-code
// dispatch and spatial (up/down/left/right) neighbor search, the other piece this TD's own
// action items call out as "the most testable without a real DOM/TV remote". Runs the real,
// unmodified file via Node's built-in vm module against a fake document/element stub.

const test = require('node:test');
const assert = require('node:assert/strict');
const { createFakeElement, createFakeDocument, runAppFile } = require('./dom_stub');

function makeNav(registry) {
    const document = createFakeDocument(registry || {});
    const context = runAppFile('tizen-metrics-app/js/navigation.js', { document });
    const nav = new context.window.TVNav();
    return { nav, document, KEY: context.window.TV_KEY };
}

function fakeKeyEvent(keyCode) {
    let prevented = false;
    return {
        keyCode,
        preventDefault() {
            prevented = true;
        },
        get defaultPrevented() {
            return prevented;
        },
    };
}

test('TVNav: constructor registers a single keydown listener on document', () => {
    const { document } = makeNav();
    assert.equal((document._listeners.keydown || []).length, 1);
});

test('TVNav: TV_KEY exposes the documented Samsung remote key codes', () => {
    const { KEY } = makeNav();
    assert.equal(KEY.UP, 38);
    assert.equal(KEY.DOWN, 40);
    assert.equal(KEY.LEFT, 37);
    assert.equal(KEY.RIGHT, 39);
    assert.equal(KEY.OK, 13);
    assert.equal(KEY.BACK, 10009);
    assert.equal(KEY.EXIT, 10182);
    assert.equal(KEY.RED, 403);
    assert.equal(KEY.GREEN, 404);
    assert.equal(KEY.YELLOW, 405);
    assert.equal(KEY.BLUE, 406);
});

test('TVNav: OK key emits "ok" with the focused element and clicks it', () => {
    const { nav, KEY } = makeNav();
    const el = createFakeElement({ rect: { left: 0, top: 0, width: 10, height: 10 } });
    nav._setFocus(el);

    let emitted = null;
    let clicked = false;
    nav.on('ok', (e) => {
        emitted = e;
    });
    el.addEventListener('click', () => {
        clicked = true;
    });

    const evt = fakeKeyEvent(KEY.OK);
    nav._onKeyDown(evt);

    assert.equal(emitted, el);
    assert.ok(clicked, 'OK must call the focused element\'s native .click()');
    assert.ok(evt.defaultPrevented);
});

test('TVNav: BACK key emits "back" with the currently focused element', () => {
    const { nav, KEY } = makeNav();
    const el = createFakeElement({ rect: { left: 0, top: 0, width: 10, height: 10 } });
    nav._setFocus(el);
    let emitted = 'unset';
    nav.on('back', (e) => {
        emitted = e;
    });
    nav._onKeyDown(fakeKeyEvent(KEY.BACK));
    assert.equal(emitted, el);
});

for (const [name, code] of [
    ['blue', 406],
    ['green', 404],
    ['red', 403],
    ['yellow', 405],
    ['info', 457],
]) {
    test('TVNav: ' + name.toUpperCase() + ' key emits "' + name + '" with null payload', () => {
        const { nav } = makeNav();
        let called = false;
        let payload = 'unset';
        nav.on(name, (e) => {
            called = true;
            payload = e;
        });
        nav._onKeyDown(fakeKeyEvent(code));
        assert.ok(called);
        assert.equal(payload, null);
    });
}

test('TVNav: EXIT key emits "exit" and swallows the ReferenceError from a missing tizen global', () => {
    // Outside a real Tizen WebKit runtime there is no global `tizen` object — the handler's own
    // try/catch around tizen.application.getCurrentApplication().exit() must swallow that
    // ReferenceError rather than letting it propagate out of _onKeyDown().
    const { nav, KEY } = makeNav();
    let emitted = 'unset';
    nav.on('exit', (e) => {
        emitted = e;
    });
    assert.doesNotThrow(() => nav._onKeyDown(fakeKeyEvent(KEY.EXIT)));
    assert.equal(emitted, null);
});

test('TVNav: an unrecognized key code emits nothing and does not throw', () => {
    const { nav } = makeNav();
    let calledAnything = false;
    ['ok', 'back', 'exit', 'blue', 'green', 'red', 'yellow', 'info'].forEach((ev) =>
        nav.on(ev, () => {
            calledAnything = true;
        })
    );
    assert.doesNotThrow(() => nav._onKeyDown(fakeKeyEvent(999999)));
    assert.equal(calledAnything, false);
});

test('TVNav.refresh: filters focusable elements by offsetWidth/offsetHeight, focuses the first', () => {
    const visible = createFakeElement({
        id: 'a',
        offsetSize: { width: 100, height: 20 },
        rect: { left: 0, top: 0, width: 100, height: 20 },
    });
    visible._isFocusable = true;
    const hidden = createFakeElement({
        id: 'b',
        offsetSize: { width: 0, height: 0 },
        rect: { left: 0, top: 0, width: 0, height: 0 },
    });
    hidden._isFocusable = true;

    const { nav } = makeNav({ a: visible, b: hidden });
    nav.refresh();

    assert.equal(nav._focusables.length, 1);
    assert.equal(nav._focusables[0], visible);
    assert.equal(nav._focused, visible, 'refresh() must auto-focus the first focusable element');
    assert.ok(visible.classList.contains('focused'));
});

test('TVNav.focusById: focuses the element with the given id via document.getElementById', () => {
    const el = createFakeElement({ id: 'target', rect: { left: 0, top: 0, width: 10, height: 10 } });
    const { nav } = makeNav({ target: el });
    nav.focusById('target');
    assert.equal(nav._focused, el);
    assert.ok(el.classList.contains('focused'));
});

test('TVNav.focusById: a missing id is a no-op, not a crash', () => {
    const { nav } = makeNav({});
    assert.doesNotThrow(() => nav.focusById('does-not-exist'));
    assert.equal(nav._focused, null);
});

test('TVNav._findNeighbor: picks the nearest element strictly in the requested direction', () => {
    const { nav } = makeNav();
    const center = createFakeElement({ rect: { left: 100, top: 100, width: 20, height: 20 } });
    const below = createFakeElement({ rect: { left: 100, top: 200, width: 20, height: 20 } });
    const above = createFakeElement({ rect: { left: 100, top: 0, width: 20, height: 20 } });
    const rightEl = createFakeElement({ rect: { left: 300, top: 100, width: 20, height: 20 } });
    nav._focusables = [center, below, above, rightEl];

    assert.equal(nav._findNeighbor(center, 'down'), below);
    assert.equal(nav._findNeighbor(center, 'up'), above);
    assert.equal(nav._findNeighbor(center, 'right'), rightEl);
    assert.equal(nav._findNeighbor(center, 'left'), null, 'nothing to the left should return null');
});

test('TVNav._findNeighbor: prefers the closer of two candidates in the same direction', () => {
    const { nav } = makeNav();
    const from = createFakeElement({ rect: { left: 100, top: 100, width: 20, height: 20 } });
    const near = createFakeElement({ rect: { left: 100, top: 150, width: 20, height: 20 } });
    const far = createFakeElement({ rect: { left: 100, top: 400, width: 20, height: 20 } });
    nav._focusables = [from, near, far];
    assert.equal(nav._findNeighbor(from, 'down'), near);
});

test('TVNav._findNeighbor: penalizes off-axis distance more heavily than the requested axis', () => {
    const { nav } = makeNav();
    const from = createFakeElement({ rect: { left: 100, top: 100, width: 20, height: 20 } });
    // Directly below, moderately far.
    const directlyBelow = createFakeElement({ rect: { left: 100, top: 300, width: 20, height: 20 } });
    // Slightly closer in the primary (y) axis but far off to the side — score should lose to
    // directlyBelow because secondary-axis distance is weighted 2.5x.
    const offToTheSide = createFakeElement({
        rect: { left: 500, top: 280, width: 20, height: 20 },
    });
    nav._focusables = [from, directlyBelow, offToTheSide];
    assert.equal(nav._findNeighbor(from, 'down'), directlyBelow);
});

test('TVNav._move: with nothing focused yet, falls back to focusing the first focusable', () => {
    const el = createFakeElement({
        id: 'a',
        offsetSize: { width: 10, height: 10 },
        rect: { left: 0, top: 0, width: 10, height: 10 },
    });
    el._isFocusable = true;
    const { nav } = makeNav({ a: el });
    nav._focusables = [];
    nav._focused = null;
    // _move() itself calls _focusFirst(), which needs _focusables already populated —
    // populate it directly (bypassing refresh()/its DOM query) to isolate _move()'s own logic.
    nav._focusables = [el];
    nav._move('down');
    assert.equal(nav._focused, el);
});

test('TVNav.destroy: removes the keydown listener it registered', () => {
    const { nav, document } = makeNav();
    assert.equal((document._listeners.keydown || []).length, 1);
    nav.destroy();
    assert.equal((document._listeners.keydown || []).length, 0);
});
