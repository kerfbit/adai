'use strict';

// Minimal, dependency-free DOM/canvas stubs for testing tizen-metrics-app/js/*.js (TD-049).
// No jsdom or any other npm package is installed or otherwise part of this project's toolchain —
// this mirrors the C++ side's own "no bats-core, no pytest" stance (see tests/scripts/harness.sh)
// by using only Node's built-in `vm` module plus these hand-written stubs, covering exactly the
// DOM/canvas surface the three app files actually call (confirmed by reading each file; not a
// general-purpose DOM shim).
//
// The real, unmodified app files are executed via vm.runInContext() against this stub — this
// tests the actual shipped source, not a reimplementation of its logic.

const vm = require('vm');
const fs = require('fs');

class ClassList {
    constructor() {
        this._set = new Set();
    }
    add() {
        for (const c of arguments) this._set.add(c);
    }
    remove() {
        for (const c of arguments) this._set.delete(c);
    }
    contains(c) {
        return this._set.has(c);
    }
    toggle(c, force) {
        const has = this._set.has(c);
        const want = force === undefined ? !has : !!force;
        if (want) this._set.add(c);
        else this._set.delete(c);
        return want;
    }
}

// A fake element supporting exactly what navigation.js/app.js touch: geometry
// (getBoundingClientRect, offsetWidth/offsetHeight — settable via `rect`/`offsetSize` in
// `attrs`), classList, style, textContent/innerHTML, get/setAttribute, focus/click,
// addEventListener/removeEventListener (recorded, dispatchable via fireEvent below),
// appendChild/querySelectorAll (only as needed by the tests that use them).
function createFakeElement(attrs) {
    attrs = attrs || {};
    const el = {
        id: attrs.id || '',
        tagName: attrs.tagName || 'DIV',
        className: '',
        textContent: '',
        innerHTML: '',
        tabIndex: -1,
        style: {},
        classList: new ClassList(),
        _attrs: Object.assign({}, attrs.dataset || {}),
        _listeners: {},
        _rect: attrs.rect || { left: 0, top: 0, width: 0, height: 0 },
        offsetWidth: attrs.offsetSize ? attrs.offsetSize.width : (attrs.rect ? attrs.rect.width : 0),
        offsetHeight: attrs.offsetSize ? attrs.offsetSize.height : (attrs.rect ? attrs.rect.height : 0),
        _children: [],
        getBoundingClientRect() {
            return this._rect;
        },
        getAttribute(name) {
            return Object.prototype.hasOwnProperty.call(this._attrs, name) ? this._attrs[name] : null;
        },
        setAttribute(name, value) {
            this._attrs[name] = String(value);
        },
        addEventListener(type, cb) {
            (this._listeners[type] = this._listeners[type] || []).push(cb);
        },
        removeEventListener(type, cb) {
            const list = this._listeners[type];
            if (!list) return;
            const idx = list.indexOf(cb);
            if (idx !== -1) list.splice(idx, 1);
        },
        dispatch(type, evt) {
            const list = this._listeners[type] || [];
            list.slice().forEach((cb) => cb(evt));
        },
        focus() {
            this._focused = true;
        },
        click() {
            this.dispatch('click', {});
        },
        appendChild(child) {
            this._children.push(child);
            return child;
        },
        querySelectorAll() {
            return [];
        },
    };
    return el;
}

// A fake canvas 2D context recording every call (method name + args) into `.calls`, so a test
// can assert on the exact coordinates chart.js's draw() computed and passed to the real canvas
// API — this is chart.js's actual coordinate math under test, not a reimplementation of it.
function createFakeCanvasContext() {
    const ctx = { calls: [] };
    const methods = [
        'clearRect', 'beginPath', 'moveTo', 'lineTo', 'stroke', 'fill', 'closePath',
        'bezierCurveTo', 'arc', 'fillText', 'scale',
    ];
    methods.forEach((name) => {
        ctx[name] = function () {
            ctx.calls.push({ method: name, args: Array.prototype.slice.call(arguments) });
        };
    });
    // Settable properties chart.js assigns (strokeStyle, fillStyle, lineWidth, font,
    // textAlign, textBaseline) — plain read/write, no recording needed for these.
    return ctx;
}

function createFakeCanvas(ctx, rect) {
    return {
        _ctx: ctx,
        width: 0,
        height: 0,
        getContext() {
            return this._ctx;
        },
        getBoundingClientRect() {
            return rect || { width: 580, height: 200 };
        },
    };
}

// A fake document: getElementById resolves from a plain registry object the test populates,
// addEventListener/removeEventListener record document-level listeners (navigation.js's own
// keydown handler), contains() always true unless explicitly told otherwise (only used by
// TVNav.refresh()'s "is the focused element still attached" check, not exercised by these tests
// beyond the always-true default).
function createFakeDocument(registry) {
    registry = registry || {};
    const doc = {
        _registry: registry,
        _listeners: {},
        getElementById(id) {
            return Object.prototype.hasOwnProperty.call(registry, id) ? registry[id] : null;
        },
        querySelectorAll(selector) {
            if (selector === '.focusable') {
                return Object.keys(registry)
                    .map((k) => registry[k])
                    .filter((el) => el && el._isFocusable);
            }
            return [];
        },
        createElement() {
            return createFakeElement();
        },
        addEventListener(type, cb) {
            (doc._listeners[type] = doc._listeners[type] || []).push(cb);
        },
        removeEventListener(type, cb) {
            const list = doc._listeners[type];
            if (!list) return;
            const idx = list.indexOf(cb);
            if (idx !== -1) list.splice(idx, 1);
        },
        contains() {
            return true;
        },
        dispatch(type, evt) {
            (doc._listeners[type] || []).slice().forEach((cb) => cb(evt));
        },
    };
    return doc;
}

// Loads `relPath` (relative to the repo root) and runs it in a vm context pre-populated with
// `window`/`document`/whatever else `extraGlobals` supplies, returning that context so the
// test can read off what the file exported onto `window` (e.g. context.window.LossChart).
function runAppFile(relPath, extraGlobals) {
    const repoRoot = require('path').resolve(__dirname, '..', '..');
    const fullPath = require('path').join(repoRoot, relPath);
    const source = fs.readFileSync(fullPath, 'utf8');

    const window = Object.assign({ devicePixelRatio: 1 }, extraGlobals && extraGlobals.window);
    const context = Object.assign(
        {
            window,
            console,
        },
        extraGlobals
    );
    context.window = window;
    vm.createContext(context);
    vm.runInContext(source, context, { filename: fullPath });
    return context;
}

// Extracts a single top-level `function <name>(...) { ... }` declaration's exact source text
// from `source` by walking brace depth from the opening `{` — used for app.js, which (unlike
// chart.js/navigation.js) never exports its pure helper functions onto `window`. This tests the
// real, unmodified function body actually shipped in the file, not a reimplementation of its
// logic; only used for genuinely standalone helpers that don't close over app.js's own
// Config/State/UI module state (see app_test.js for which ones qualify and why).
function extractFunction(source, name) {
    const marker = 'function ' + name + '(';
    const start = source.indexOf(marker);
    if (start === -1) {
        throw new Error('extractFunction: no "function ' + name + '(" found');
    }
    const braceStart = source.indexOf('{', start);
    if (braceStart === -1) {
        throw new Error('extractFunction: no opening brace found for ' + name);
    }
    let depth = 0;
    for (let i = braceStart; i < source.length; i++) {
        if (source[i] === '{') depth++;
        else if (source[i] === '}') {
            depth--;
            if (depth === 0) {
                return source.slice(start, i + 1);
            }
        }
    }
    throw new Error('extractFunction: unbalanced braces for ' + name);
}

// Extracts one or more functions from `relPath` and evaluates them in a context carrying
// `extraGlobals` (e.g. a fake `document` for escapeHtml), returning an object mapping each name
// to the live function.
function extractFunctions(relPath, names, extraGlobals) {
    const repoRoot = require('path').resolve(__dirname, '..', '..');
    const fullPath = require('path').join(repoRoot, relPath);
    const source = fs.readFileSync(fullPath, 'utf8');

    const context = Object.assign({ console }, extraGlobals);
    vm.createContext(context);

    const bodies = names.map((n) => extractFunction(source, n)).join('\n');
    const resultExpr = '({' + names.map((n) => n + ': ' + n).join(', ') + '})';
    return vm.runInContext('(function(){' + bodies + '\nreturn ' + resultExpr + ';})()', context,
                           { filename: fullPath });
}

module.exports = {
    ClassList,
    createFakeElement,
    createFakeCanvasContext,
    createFakeCanvas,
    createFakeDocument,
    runAppFile,
    extractFunction,
    extractFunctions,
};
