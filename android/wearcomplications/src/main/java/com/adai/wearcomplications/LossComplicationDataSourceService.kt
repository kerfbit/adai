package com.adai.wearcomplications

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-07


class LossComplicationDataSourceService : TrainingComplicationDataSourceService() {
    override val label: String = "Loss"
    override fun currentValue(snapshot: TrainingSnapshot): Double = snapshot.currentLoss
    override fun minValue(snapshot: TrainingSnapshot): Double = snapshot.lossMin
    override fun maxValue(snapshot: TrainingSnapshot): Double = snapshot.lossMax
}
