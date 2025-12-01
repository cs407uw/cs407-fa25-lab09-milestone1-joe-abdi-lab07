package com.cs407.lab09

import android.hardware.Sensor
import android.hardware.SensorEvent
import androidx.compose.ui.geometry.Offset
import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update

class BallViewModel : ViewModel() {

    private var ball: Ball? = null
    private var lastTimestamp = 0L

    private val _ballPosition = MutableStateFlow(Offset.Zero)
    val ballPosition: StateFlow<Offset> = _ballPosition

    fun initBall(fieldWidth: Float, fieldHeight: Float, ballSizePx: Float) {
        if (ball == null) {
            ball = Ball(fieldWidth, fieldHeight, ballSizePx)
            _ballPosition.value = Offset(ball!!.posX, ball!!.posY)
        }
    }

    fun onSensorDataChanged(event: SensorEvent) {
        val b = ball ?: return

        if (event.sensor.type == Sensor.TYPE_GRAVITY) {
            if (lastTimestamp != 0L) {
                val dt = (event.timestamp - lastTimestamp) * (1f / 1_000_000_000f)

                val xAcc = event.values[0]
                val yAcc = -event.values[1]

                b.updatePositionAndVelocity(xAcc, yAcc, dt)

                _ballPosition.update { Offset(b.posX, b.posY) }
            }

            lastTimestamp = event.timestamp
        }
    }

    fun reset() {
        ball?.reset()
        ball?.let { _ballPosition.value = Offset(it.posX, it.posY) }
        lastTimestamp = 0L
    }
}

