'use strict';

// Tests for tizen-metrics-app/js/chart.js (TD-049) — LossChart's coordinate math, the piece this
// TD's own action items call out as "the most testable without a real DOM/TV remote". Runs the
// real, unmodified file via Node's built-in vm module against a fake canvas 2D context that
// records every call, so assertions check the actual coordinates draw() computes and passes to
// the canvas API — not a reimplementation of xFor()/yFor()'s formulas.

const test = require('node:test');
const assert = require('node:assert/strict');
const {
    createFakeCanvasContext,
    createFakeCanvas,
    createFakeDocument,
    runAppFile,
} = require('./dom_stub');

function makeChart(rect) {
    const ctx = createFakeCanvasContext();
    const canvas = createFakeCanvas(ctx, rect);
    const document = createFakeDocument({ 'loss-chart': canvas });
    const context = runAppFile('tizen-metrics-app/js/chart.js', { document });
    const chart = new context.window.LossChart('loss-chart');
    return { chart, ctx };
}

function calls(ctx, method) {
    return ctx.calls.filter((c) => c.method === method);
}

function hasCallNear(ctx, method, x, y, tol) {
    tol = tol === undefined ? 0.01 : tol;
    return calls(ctx, method).some(
        (c) => Math.abs(c.args[0] - x) < tol && Math.abs(c.args[1] - y) < tol
    );
}

test('LossChart: constructor sizes the canvas from the element rect and devicePixelRatio', () => {
    const { chart } = makeChart({ width: 580, height: 200 });
    assert.equal(chart._w, 580);
    assert.equal(chart._h, 200);
});

test('LossChart: draw() with no data shows the empty-state message, draws nothing else', () => {
    const { chart, ctx } = makeChart();
    chart.update([], []);
    const fillTextCalls = calls(ctx, 'fillText');
    assert.equal(fillTextCalls.length, 1, 'only the empty-state text should be drawn');
    assert.equal(fillTextCalls[0].args[0], 'No epoch data yet');
    assert.equal(calls(ctx, 'arc').length, 0, 'no data points should be plotted');
});

test('LossChart: draw() plots points at the exact coordinates xFor()/yFor() compute', () => {
    const { chart, ctx } = makeChart({ width: 580, height: 200 });
    chart.update([1, 2], []);

    // pad = {top:20,left:64,...}; plotW = 580-64-30 = 486; plotH = 200-20-48 = 132.
    // range [1,2] padded by 12% -> [0.88, 2.12], range 1.24.
    const expectedX0 = 64;
    const expectedX1 = 64 + 486;
    const expectedY0 = 20 + 132 - ((1 - 0.88) / 1.24) * 132;
    const expectedY1 = 20 + 132 - ((2 - 0.88) / 1.24) * 132;

    assert.ok(
        hasCallNear(ctx, 'arc', expectedX0, expectedY0),
        'expected a point at (' + expectedX0 + ', ' + expectedY0.toFixed(3) + ')'
    );
    assert.ok(
        hasCallNear(ctx, 'arc', expectedX1, expectedY1),
        'expected a point at (' + expectedX1 + ', ' + expectedY1.toFixed(3) + ')'
    );
});

test('LossChart: a series with only one valid point draws no line/area/points for it', () => {
    const { chart, ctx } = makeChart({ width: 580, height: 200 });
    chart.update([5], []);
    // drawSeries() returns early when validPoints.length < 2 — the single-point train series
    // should contribute no arc/bezierCurveTo calls, even though grid/axis drawing still happens.
    assert.equal(calls(ctx, 'arc').length, 0);
    assert.equal(calls(ctx, 'bezierCurveTo').length, 0);
});

test('LossChart: non-finite values (NaN/Infinity) are filtered out of the plotted range', () => {
    const { chart, ctx } = makeChart({ width: 580, height: 200 });
    chart.update([1, NaN, 2, Infinity], []);
    // Only the two finite values (1, 2) should count toward validPoints — same math as the
    // two-point case above.
    const expectedY0 = 20 + 132 - ((1 - 0.88) / 1.24) * 132;
    assert.ok(hasCallNear(ctx, 'arc', 64, expectedY0));
});

test('LossChart: custom xLabels are used instead of the default "E<n>" labels', () => {
    const { chart, ctx } = makeChart({ width: 580, height: 200 });
    chart.update([1, 2], [], ['step-1', 'step-2']);
    const texts = calls(ctx, 'fillText').map((c) => c.args[0]);
    assert.ok(texts.includes('step-1'), 'custom label for point 1 should be drawn');
    assert.ok(texts.includes('step-2'), 'custom label for point 2 should be drawn');
    assert.ok(!texts.includes('E1'), 'default "E1" label must not appear when xLabels is given');
});

test('LossChart: omitting xLabels falls back to default "E<n>" epoch labels', () => {
    const { chart, ctx } = makeChart({ width: 580, height: 200 });
    chart.update([1, 2], []);
    const texts = calls(ctx, 'fillText').map((c) => c.args[0]);
    assert.ok(texts.includes('E1'));
    assert.ok(texts.includes('E2'));
});
