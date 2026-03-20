package com.bunoraa.admin.benchmark

import androidx.benchmark.macro.CompilationMode
import androidx.benchmark.macro.FrameTimingMetric
import androidx.benchmark.macro.MacrobenchmarkRule
import androidx.benchmark.macro.junit4.measureRepeated
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import androidx.test.uiautomator.Direction
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.UiSelector
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
@LargeTest
class ScrollBenchmarks {
    @get:Rule
    val benchmarkRule = MacrobenchmarkRule()

    @Test
    fun dashboardScroll() = benchmarkRule.measureRepeated(
        packageName = "com.bunoraa.admin",
        metrics = listOf(FrameTimingMetric()),
        compilationMode = CompilationMode.SpeedProfile(),
        iterations = 5,
    ) {
        startActivityAndWait()
        val device = UiDevice.getInstance(InstrumentationRegistry.getInstrumentation())
        device.waitForIdle()

        // Attempt to scroll any scrollable container; fall back to manual swipes.
        val scrollable = device.findObject(UiSelector().scrollable(true))
        if (scrollable.exists()) {
            scrollable.fling(Direction.DOWN)
            scrollable.fling(Direction.UP)
        } else {
            device.swipe(
                device.displayWidth / 2,
                device.displayHeight * 3 / 4,
                device.displayWidth / 2,
                device.displayHeight / 4,
                10,
            )
            device.swipe(
                device.displayWidth / 2,
                device.displayHeight / 4,
                device.displayWidth / 2,
                device.displayHeight * 3 / 4,
                10,
            )
        }
    }
}
