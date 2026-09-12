'use strict';

// Tests for tizen-metrics-app/js/app.js's pure helper functions (TD-049). Unlike chart.js/
// navigation.js, app.js exports nothing onto `window` and its whole IIFE executes top-level
// browser calls (localStorage, document.getElementById for every UI element, setInterval,
// document.addEventListener('DOMContentLoaded', ...)) the moment it loads — running the entire
// file would need a much larger stub for very little payoff, since none of that startup wiring
// is pure logic. Instead, this extracts the real, unmodified source of specific standalone
// helper functions (see extractFunctions in dom_stub.js — walks brace depth from the real file,
// not a reimplementation) that don't close over app.js's own Config/State/UI module state, and
// evaluates just those against a minimal stub.

const test = require('node:test');
const assert = require('node:assert/strict');
const { createFakeElement, extractFunctions } = require('./dom_stub');

const APP_JS = 'tizen-metrics-app/js/app.js';

test('fmt: formats numbers to the given decimal places, defaulting to 4', () => {
    const { fmt } = extractFunctions(APP_JS, ['fmt']);
    assert.equal(fmt(1.23456), '1.2346');
    assert.equal(fmt(1.23456, 2), '1.23');
    assert.equal(fmt(0, 2), '0.00');
});

test('fmt: null/undefined/NaN all render as the em-dash placeholder', () => {
    const { fmt } = extractFunctions(APP_JS, ['fmt']);
    assert.equal(fmt(null), '—');
    assert.equal(fmt(undefined), '—');
    assert.equal(fmt(NaN), '—');
});

test('fmt: non-number values that are not NaN-ish pass through via String()', () => {
    // fmt()'s own isNaN(val) check uses the global isNaN, which coerces its argument first —
    // isNaN('training') is true (Number('training') is NaN), so a non-numeric string like that
    // actually hits the '—' branch, not this one. Only a value that survives the coercion (a
    // boolean, or a numeric-looking string) reaches the String(val) fallback.
    const { fmt } = extractFunctions(APP_JS, ['fmt']);
    assert.equal(fmt(true), 'true');
    assert.equal(fmt('42'), '42', 'a numeric-looking string is not NaN-ish, so it also passes through unformatted');
});

test('fmt: a non-numeric, non-coercible string reads as "no data", not the literal string', () => {
    const { fmt } = extractFunctions(APP_JS, ['fmt']);
    assert.equal(fmt('training'), '—');
});

test('fmtInt: rounds and adds thousands separators', () => {
    const { fmtInt } = extractFunctions(APP_JS, ['fmtInt']);
    assert.equal(fmtInt(1234567.6), '1,234,568');
    assert.equal(fmtInt(0), '0');
});

test('fmtInt: null/undefined/NaN render as the em-dash placeholder', () => {
    const { fmtInt } = extractFunctions(APP_JS, ['fmtInt']);
    assert.equal(fmtInt(null), '—');
    assert.equal(fmtInt(NaN), '—');
});

test('pad2: zero-pads single digits, leaves two-plus-digit numbers alone', () => {
    const { pad2 } = extractFunctions(APP_JS, ['pad2']);
    assert.equal(pad2(0), '00');
    assert.equal(pad2(9), '09');
    assert.equal(pad2(10), '10');
    assert.equal(pad2(59), '59');
});

test('fmtTime: formats seconds as Hh MMm SSs, dropping leading zero units', () => {
    const { fmtTime } = extractFunctions(APP_JS, ['fmtTime', 'pad2']);
    assert.equal(fmtTime(45), '45s');
    assert.equal(fmtTime(125), '02m 05s');
    assert.equal(fmtTime(3725), '1h 02m 05s');
});

test('fmtTime: null/undefined/NaN/negative all render as the em-dash placeholder', () => {
    const { fmtTime } = extractFunctions(APP_JS, ['fmtTime', 'pad2']);
    assert.equal(fmtTime(null), '—');
    assert.equal(fmtTime(undefined), '—');
    assert.equal(fmtTime(NaN), '—');
    assert.equal(fmtTime(-5), '—');
});

test('fmtLR: uses scientific notation below 1e-4, fixed notation (trimmed) above it', () => {
    const { fmtLR } = extractFunctions(APP_JS, ['fmtLR']);
    assert.equal(fmtLR(0.00001234), '1.234e-5');
    assert.equal(fmtLR(0.001), '0.001');
    assert.equal(fmtLR(0.0001234567), '0.0001235');
});

test('fmtLR: falsy-but-not-zero values render as the em-dash; a real zero does not', () => {
    const { fmtLR } = extractFunctions(APP_JS, ['fmtLR']);
    assert.equal(fmtLR(null), '—');
    assert.equal(fmtLR(undefined), '—');
    assert.notEqual(fmtLR(0), '—', 'a genuine learning rate of exactly 0 must not read as "no data"');
});

test('gaugeArcPath: builds an SVG arc "M x y A r r 0 large-arc 1 x y" path', () => {
    const { gaugeArcPath } = extractFunctions(APP_JS, ['gaugeArcPath']);
    // From 180deg to 0deg around (0,0) r=10: start (-10,0), end (10,-0) -- screen-space y is
    // inverted (y = cy - r*sin), and a 180-degree sweep is exactly the "large arc" boundary
    // (fromDeg - toDeg > 180 is false at exactly 180, so large-arc-flag is 0).
    const path = gaugeArcPath(0, 0, 10, 180, 0);
    assert.equal(path, 'M -10.00 -0.00 A 10 10 0 0 1 10.00 0.00');
});

test('gaugeArcPath: sets the large-arc-flag when the sweep exceeds 180 degrees', () => {
    const { gaugeArcPath } = extractFunctions(APP_JS, ['gaugeArcPath']);
    const path = gaugeArcPath(0, 0, 10, 359, 0);
    assert.ok(path.includes(' A 10 10 0 1 1 '), 'sweep > 180deg must set the large-arc flag to 1');
});

test('escapeHtml: neutralizes markup via the DOM serializer (textContent round-trip)', () => {
    const document = {
        createElement() {
            return createFakeElement();
        },
    };
    // The fake element's .innerHTML is never derived from .textContent by our stub (it doesn't
    // implement a real serializer) — so this test uses a slightly smarter local stub that does,
    // to actually exercise escapeHtml()'s real behavior (assigning textContent, reading back
    // innerHTML) rather than trivially short-circuiting.
    const el = {
        set textContent(v) {
            this._text = v;
        },
        get textContent() {
            return this._text;
        },
        get innerHTML() {
            return String(this._text)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');
        },
    };
    document.createElement = () => el;
    const { escapeHtml } = extractFunctions(APP_JS, ['escapeHtml'], { document });
    assert.equal(escapeHtml('<img src=x onerror="alert(1)">'),
                '&lt;img src=x onerror="alert(1)"&gt;');
    assert.equal(escapeHtml('plain text'), 'plain text');
});

test('applyRangeColor: green below warn, amber below error, red at/above error', () => {
    const { applyRangeColor } = extractFunctions(APP_JS, ['applyRangeColor']);
    const el = createFakeElement();
    applyRangeColor(el, 1.0, 2.0, 4.0);
    assert.ok(el.classList.contains('val-good'));

    applyRangeColor(el, 3.0, 2.0, 4.0);
    assert.ok(el.classList.contains('val-warn'));
    assert.ok(!el.classList.contains('val-good'));

    applyRangeColor(el, 5.0, 2.0, 4.0);
    assert.ok(el.classList.contains('val-error'));
    assert.ok(!el.classList.contains('val-warn'));
});

test('applyRangeColor: null/NaN clears all color classes and sets none', () => {
    const { applyRangeColor } = extractFunctions(APP_JS, ['applyRangeColor']);
    const el = createFakeElement();
    el.classList.add('val-good');
    applyRangeColor(el, NaN, 2.0, 4.0);
    assert.ok(!el.classList.contains('val-good'));
    assert.ok(!el.classList.contains('val-warn'));
    assert.ok(!el.classList.contains('val-error'));
});

test('applyGenQualColor: higher BLEU/ROUGE is better (>=0.3 good, >=0.1 warn, else error)', () => {
    const { applyGenQualColor } = extractFunctions(APP_JS, ['applyGenQualColor']);
    const el = createFakeElement();
    applyGenQualColor(el, 0.35);
    assert.ok(el.classList.contains('val-good'));
    applyGenQualColor(el, 0.15);
    assert.ok(el.classList.contains('val-warn'));
    applyGenQualColor(el, 0.05);
    assert.ok(el.classList.contains('val-error'));
});

test('applyLRColor: the healthy learning-rate band is [1e-5, 1e-3]', () => {
    const { applyLRColor } = extractFunctions(APP_JS, ['applyLRColor']);
    const el = createFakeElement();
    applyLRColor(el, 1e-4);
    assert.ok(el.classList.contains('val-good'));
    applyLRColor(el, 5e-3);
    assert.ok(el.classList.contains('val-warn'));
    applyLRColor(el, 5e-2);
    assert.ok(el.classList.contains('val-error'));
});
