/* ============================================================
   chart.js — Canvas-based loss chart renderer
   ADAI Training Metrics Dashboard
   ============================================================ */

(function(window) {
    'use strict';

    /**
     * LossChart — draws epoch-by-epoch training/validation loss curves
     * on a <canvas> element using the 2D canvas API.
     */
    function LossChart(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx    = this.canvas.getContext('2d');

        /* Colors (must match CSS vars) */
        this.colors = {
            trainLine:   '#58a6ff',
            valLine:     '#3fb950',
            trainFill:   'rgba(88,166,255,0.08)',
            valFill:     'rgba(63,185,80,0.08)',
            grid:        'rgba(255,255,255,0.06)',
            axis:        'rgba(255,255,255,0.15)',
            label:       '#8b949e',
            crosshair:   'rgba(255,255,255,0.25)',
            pointTrain:  '#58a6ff',
            pointVal:    '#3fb950',
        };

        this.padding = { top: 20, right: 30, bottom: 48, left: 64 };

        /* History: arrays of numbers */
        this.trainLosses = [];
        this.valLosses   = [];

        /* Track canvas size for HiDPI */
        this._dpr = window.devicePixelRatio || 1;
        this._resize();
    }

    /** Update canvas internal resolution to match display size. */
    LossChart.prototype._resize = function() {
        var rect = this.canvas.getBoundingClientRect();
        var w = rect.width  || 580;
        var h = rect.height || 200;
        var dpr = this._dpr;

        this.canvas.width  = w * dpr;
        this.canvas.height = h * dpr;
        this.ctx.scale(dpr, dpr);

        this._w = w;
        this._h = h;
    };

    /** Set new data and redraw. */
    LossChart.prototype.update = function(trainLosses, valLosses) {
        this.trainLosses = trainLosses || [];
        this.valLosses   = valLosses   || [];
        this.draw();
    };

    /** Clear and redraw everything. */
    LossChart.prototype.draw = function() {
        var ctx  = this.ctx;
        var w    = this._w;
        var h    = this._h;
        var pad  = this.padding;
        var c    = this.colors;

        ctx.clearRect(0, 0, w, h);

        var plotW = w - pad.left - pad.right;
        var plotH = h - pad.top  - pad.bottom;

        /* Find data range */
        var allVals = this.trainLosses.concat(this.valLosses).filter(function(v) {
            return typeof v === 'number' && isFinite(v);
        });

        if (allVals.length === 0) {
            this._drawEmpty(ctx, w, h);
            return;
        }

        var minV = Math.min.apply(null, allVals);
        var maxV = Math.max.apply(null, allVals);
        var range = maxV - minV || 1;
        var padV  = range * 0.12;
        minV -= padV;
        maxV += padV;
        range = maxV - minV;

        var numPoints = Math.max(this.trainLosses.length, this.valLosses.length, 2);

        /* helpers */
        var self = this;
        function xFor(i) {
            return pad.left + (i / (numPoints - 1)) * plotW;
        }
        function yFor(v) {
            return pad.top + plotH - ((v - minV) / range) * plotH;
        }

        /* --- Grid --- */
        var gridLines = 5;
        ctx.strokeStyle = c.grid;
        ctx.lineWidth   = 1;

        for (var gi = 0; gi <= gridLines; gi++) {
            var gy = pad.top + (gi / gridLines) * plotH;
            ctx.beginPath();
            ctx.moveTo(pad.left, gy);
            ctx.lineTo(pad.left + plotW, gy);
            ctx.stroke();

            /* Y-axis label */
            var labelV = maxV - (gi / gridLines) * range;
            ctx.fillStyle = c.label;
            ctx.font      = '18px monospace';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            ctx.fillText(labelV.toFixed(3), pad.left - 8, gy);
        }

        /* X-axis epoch labels */
        ctx.fillStyle    = c.label;
        ctx.font         = '18px monospace';
        ctx.textAlign    = 'center';
        ctx.textBaseline = 'top';

        var maxXLabels = Math.min(numPoints, 10);
        var step = Math.ceil(numPoints / maxXLabels);
        for (var xi = 0; xi < numPoints; xi += step) {
            var xx = xFor(xi);
            ctx.fillText('E' + xi, xx, pad.top + plotH + 8);
        }

        /* --- Axes --- */
        ctx.strokeStyle = c.axis;
        ctx.lineWidth   = 1.5;
        ctx.beginPath();
        ctx.moveTo(pad.left, pad.top);
        ctx.lineTo(pad.left, pad.top + plotH);
        ctx.lineTo(pad.left + plotW, pad.top + plotH);
        ctx.stroke();

        /* --- Draw filled area + line for each series --- */
        function drawSeries(losses, lineColor, fillColor, pointColor) {
            if (!losses || losses.length === 0) return;

            var validPoints = [];
            for (var p = 0; p < losses.length; p++) {
                if (typeof losses[p] === 'number' && isFinite(losses[p])) {
                    validPoints.push({ x: xFor(p), y: yFor(losses[p]) });
                }
            }
            if (validPoints.length < 2) return;

            /* Filled area */
            ctx.beginPath();
            ctx.moveTo(validPoints[0].x, pad.top + plotH);
            for (var fp = 0; fp < validPoints.length; fp++) {
                ctx.lineTo(validPoints[fp].x, validPoints[fp].y);
            }
            ctx.lineTo(validPoints[validPoints.length - 1].x, pad.top + plotH);
            ctx.closePath();
            ctx.fillStyle = fillColor;
            ctx.fill();

            /* Line */
            ctx.beginPath();
            ctx.moveTo(validPoints[0].x, validPoints[0].y);
            for (var lp = 1; lp < validPoints.length; lp++) {
                /* smooth with bezier */
                var prev = validPoints[lp - 1];
                var curr = validPoints[lp];
                var cp1x = prev.x + (curr.x - prev.x) * 0.4;
                var cp2x = curr.x - (curr.x - prev.x) * 0.4;
                ctx.bezierCurveTo(cp1x, prev.y, cp2x, curr.y, curr.x, curr.y);
            }
            ctx.strokeStyle = lineColor;
            ctx.lineWidth   = 2.5;
            ctx.stroke();

            /* Points */
            for (var pp = 0; pp < validPoints.length; pp++) {
                ctx.beginPath();
                ctx.arc(validPoints[pp].x, validPoints[pp].y, 4, 0, Math.PI * 2);
                ctx.fillStyle = pointColor;
                ctx.fill();
            }
        }

        drawSeries(this.trainLosses, c.trainLine, c.trainFill, c.pointTrain);
        drawSeries(this.valLosses,   c.valLine,   c.valFill,   c.pointVal);
    };

    /** Draw empty state message. */
    LossChart.prototype._drawEmpty = function(ctx, w, h) {
        ctx.fillStyle    = 'rgba(255,255,255,0.1)';
        ctx.font         = '24px sans-serif';
        ctx.textAlign    = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('No epoch data yet', w / 2, h / 2);
    };

    window.LossChart = LossChart;

})(window);
