package com.bunoraa.admin.benchmark

import androidx.benchmark.macro.CompilationMode
import androidx.benchmark.macro.MacrobenchmarkRule
import androidx.benchmark.macro.StartupMode
import androidx.benchmark.macro.StartupTimingMetric
import androidx.benchmark.macro.junit4.MeasurementMode
import androidx.benchmark.macro.junit4.measureRepeated
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
@LargeTest
class StartupBenchmarks {
    @get:Rule
    val benchmarkRule = MacrobenchmarkRule()

    @Test
    fun startupCold() = benchmarkRule.measureRepeated(
        packageName = "com.bunoraa.admin",
        metrics = listOf(StartupTimingMetric()),
        compilationMode = CompilationMode.SpeedProfile(),
        startupMode = StartupMode.COLD,
        iterations = 5,
        measurementMode = MeasurementMode.SingleShot,
    ) {
        pressHome()
        startActivityAndWait()
    }

    @Test
    fun startupWarm() = benchmarkRule.measureRepeated(
        packageName = "com.bunoraa.admin",
        metrics = listOf(StartupTimingMetric()),
        compilationMode = CompilationMode.SpeedProfile(),
        startupMode = StartupMode.WARM,
        iterations = 5,
        measurementMode = MeasurementMode.SingleShot,
    ) {
        pressHome()
        startActivityAndWait()
    }
}
