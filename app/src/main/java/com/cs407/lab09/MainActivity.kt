package com.cs407.lab09

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.paint
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.cs407.lab09.ui.theme.Lab09Theme
import kotlin.math.roundToInt

class MainActivity : ComponentActivity() {

    private val viewModel: BallViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            Lab09Theme {
                GameScreen(viewModel)
            }
        }
    }
}

@Composable
fun GameScreen(viewModel: BallViewModel) {

    val context = LocalContext.current
    val sensorManager = remember {
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    }
    val gravitySensor = remember {
        sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY)
    }

    DisposableEffect(sensorManager, gravitySensor) {

        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent?) {
                event?.let { e ->
                    viewModel.onSensorDataChanged(e)
                }
            }
            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
        }

        if (gravitySensor != null) {
            sensorManager.registerListener(
                listener,
                gravitySensor,
                SensorManager.SENSOR_DELAY_GAME
            )
        }

        onDispose {
            if (gravitySensor != null) {
                sensorManager.unregisterListener(listener)
            }
        }
    }

    val ballSize = 50.dp
    val ballSizePx = with(LocalDensity.current) { ballSize.toPx() }

    val ballPosition by viewModel.ballPosition.collectAsStateWithLifecycle()

    Column(modifier = Modifier.fillMaxSize()) {

        Button(
            onClick = { viewModel.reset() },
            modifier = Modifier
                .padding(16.dp)
                .align(Alignment.CenterHorizontally)
        ) {
            Text("Reset")
        }

        Box(
            modifier = Modifier
                .fillMaxSize()
                .weight(1f)
                .paint(
                    painterResource(id = R.drawable.field),
                    contentScale = ContentScale.FillBounds
                )
                .onSizeChanged { size ->
                    viewModel.initBall(
                        size.width.toFloat(),
                        size.height.toFloat(),
                        ballSizePx
                    )
                }
        ) {

            Image(
                painter = painterResource(id = R.drawable.soccer),
                contentDescription = "Ball",
                modifier = Modifier
                    .size(ballSize)
                    .offset {
                        IntOffset(
                            ballPosition.x.roundToInt(),
                            ballPosition.y.roundToInt()
                        )
                    }
            )
        }
    }
}

