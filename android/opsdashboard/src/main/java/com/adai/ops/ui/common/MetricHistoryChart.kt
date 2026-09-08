package com.adai.ops.ui.common

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-07


import android.graphics.Paint
import android.graphics.Typeface
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Fill
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

/**
 * Canvas-based history chart — grid, filled area, smoothed line, and epoch labels,
 * matching the look of the Tizen dashboard's LossChart (tizen-metrics-app/js/chart.js)
 * so loss and perplexity read as the same kind of chart there and here. Renders two
 * series (train vs validation) aligned by index; a full charting library would be
 * overkill for this.
 */
@Composable
fun MetricHistoryChart(
    title: String,
    trainValues: List<Double>,
    validationValues: List<Double>,
    modifier: Modifier = Modifier,
    trainLabel: String = "Train",
    validationLabel: String = "Validation",
) {
    val trainColor = MaterialTheme.colorScheme.primary
    val validationColor = MaterialTheme.colorScheme.tertiary
    val gridColor = MaterialTheme.colorScheme.outlineVariant
    val axisColor = MaterialTheme.colorScheme.outline
    val labelColor = MaterialTheme.colorScheme.onSurfaceVariant.toArgb()

    Text(title, style = MaterialTheme.typography.titleLarge)
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(16.dp)) {
        LegendDot(trainColor, trainLabel)
        LegendDot(validationColor, validationLabel)
    }

    Canvas(
        modifier = modifier
            .fillMaxWidth()
            .height(200.dp),
    ) {
        val labelPaint = Paint().apply {
            isAntiAlias = true
            typeface = Typeface.MONOSPACE
            color = labelColor
            textSize = 11.sp.toPx()
        }

        val allValues = (trainValues + validationValues).filter { it.isFinite() }
        if (allValues.isEmpty()) {
            labelPaint.textAlign = Paint.Align.CENTER
            labelPaint.textSize = 14.sp.toPx()
            drawContext.canvas.nativeCanvas.drawText(
                "No epoch data yet", size.width / 2, size.height / 2, labelPaint,
            )
            return@Canvas
        }

        val padLeft = 44.dp.toPx()
        val padRight = 8.dp.toPx()
        val padTop = 8.dp.toPx()
        val padBottom = 24.dp.toPx()
        val plotWidth = size.width - padLeft - padRight
        val plotHeight = size.height - padTop - padBottom

        var minV = allValues.min()
        var maxV = allValues.max()
        val rawRange = (maxV - minV).let { if (it > 0.0001) it else 1.0 }
        val valuePadding = rawRange * 0.12
        minV -= valuePadding
        maxV += valuePadding
        val range = maxV - minV

        val numPoints = max(max(trainValues.size, validationValues.size), 2)

        fun xFor(index: Int): Float = padLeft + (index.toFloat() / (numPoints - 1)) * plotWidth
        fun yFor(value: Double): Float = (padTop + plotHeight - ((value - minV) / range * plotHeight)).toFloat()

        // Grid lines + y-axis value labels
        val gridLines = 4
        labelPaint.textAlign = Paint.Align.RIGHT
        for (i in 0..gridLines) {
            val y = padTop + (i.toFloat() / gridLines) * plotHeight
            drawLine(gridColor, Offset(padLeft, y), Offset(padLeft + plotWidth, y), strokeWidth = 1f)
            val labelValue = maxV - (i.toFloat() / gridLines) * range
            drawContext.canvas.nativeCanvas.drawText(
                String.format(java.util.Locale.US, "%.3f", labelValue), padLeft - 6f, y + 4f, labelPaint,
            )
        }

        // X-axis epoch labels
        labelPaint.textAlign = Paint.Align.CENTER
        val maxXLabels = min(numPoints, 6)
        val step = ceil(numPoints.toFloat() / maxXLabels).toInt().coerceAtLeast(1)
        var epoch = 1
        while (epoch <= numPoints) {
            drawContext.canvas.nativeCanvas.drawText(
                "E$epoch", xFor(epoch - 1), padTop + plotHeight + 18f, labelPaint,
            )
            epoch += step
        }

        // Axes
        drawLine(axisColor, Offset(padLeft, padTop), Offset(padLeft, padTop + plotHeight), strokeWidth = 1.5f)
        drawLine(
            axisColor,
            Offset(padLeft, padTop + plotHeight),
            Offset(padLeft + plotWidth, padTop + plotHeight),
            strokeWidth = 1.5f,
        )

        fun drawSeries(values: List<Double>, lineColor: Color) {
            val points = values.mapIndexedNotNull { i, v -> if (v.isFinite()) Offset(xFor(i), yFor(v)) else null }
            if (points.size < 2) return

            val linePath = Path().apply {
                moveTo(points[0].x, points[0].y)
                for (i in 1 until points.size) {
                    val prev = points[i - 1]
                    val curr = points[i]
                    val cp1x = prev.x + (curr.x - prev.x) * 0.4f
                    val cp2x = curr.x - (curr.x - prev.x) * 0.4f
                    cubicTo(cp1x, prev.y, cp2x, curr.y, curr.x, curr.y)
                }
            }
            val fillPath = Path().apply {
                addPath(linePath)
                lineTo(points.last().x, padTop + plotHeight)
                lineTo(points.first().x, padTop + plotHeight)
                close()
            }
            drawPath(fillPath, color = lineColor.copy(alpha = 0.12f), style = Fill)
            drawPath(linePath, color = lineColor, style = androidx.compose.ui.graphics.drawscope.Stroke(width = 4f))
            points.forEach { drawCircle(color = lineColor, radius = 5f, center = it) }
        }

        drawSeries(trainValues, trainColor)
        drawSeries(validationValues, validationColor)
    }
}

@Composable
private fun LegendDot(color: Color, label: String) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        Box(
            modifier = Modifier
                .size(10.dp)
                .background(color),
        )
        Text(label, style = MaterialTheme.typography.bodyLarge)
    }
}
