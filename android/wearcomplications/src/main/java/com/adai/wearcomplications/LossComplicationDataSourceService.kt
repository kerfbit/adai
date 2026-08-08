package com.adai.wearcomplications

class LossComplicationDataSourceService : TrainingComplicationDataSourceService() {
    override val label: String = "Loss"
    override fun currentValue(snapshot: TrainingSnapshot): Double = snapshot.currentLoss
    override fun minValue(snapshot: TrainingSnapshot): Double = snapshot.lossMin
    override fun maxValue(snapshot: TrainingSnapshot): Double = snapshot.lossMax
}
