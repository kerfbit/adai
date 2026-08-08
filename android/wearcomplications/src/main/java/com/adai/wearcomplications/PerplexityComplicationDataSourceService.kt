package com.adai.wearcomplications

class PerplexityComplicationDataSourceService : TrainingComplicationDataSourceService() {
    override val label: String = "Perplexity"
    override fun currentValue(snapshot: TrainingSnapshot): Double = snapshot.currentPerplexity
    override fun minValue(snapshot: TrainingSnapshot): Double = snapshot.perplexityMin
    override fun maxValue(snapshot: TrainingSnapshot): Double = snapshot.perplexityMax
}
