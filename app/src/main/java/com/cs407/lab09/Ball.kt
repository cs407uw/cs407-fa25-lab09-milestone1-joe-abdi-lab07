package com.cs407.lab09

class Ball(
    private val backgroundWidth: Float,
    private val backgroundHeight: Float,
    private val ballSize: Float
) {

    var posX = 0f
    var posY = 0f
    var velocityX = 0f
    var velocityY = 0f
    private var accX = 0f
    private var accY = 0f
    private var isFirstUpdate = true

    init {
        reset()
    }

    fun updatePositionAndVelocity(xAcc: Float, yAcc: Float, dT: Float) {
        if (isFirstUpdate) {
            isFirstUpdate = false
            accX = xAcc
            accY = yAcc
            return
        }

        val oldAccX = accX
        val oldAccY = accY

        accX = xAcc
        accY = yAcc

        velocityX += 0.5f * (oldAccX + accX) * dT
        velocityY += 0.5f * (oldAccY + accY) * dT

        val dx = velocityX * dT + (1f/6f) * dT * dT * (3*oldAccX + accX)
        val dy = velocityY * dT + (1f/6f) * dT * dT * (3*oldAccY + accY)

        posX += dx
        posY += dy

        checkBoundaries()
    }

    fun checkBoundaries() {

        if (posX < 0f) {
            posX = 0f
            velocityX = 0f
            accX = 0f
        }

        if (posX > backgroundWidth - ballSize) {
            posX = backgroundWidth - ballSize
            velocityX = 0f
            accX = 0f
        }

        if (posY < 0f) {
            posY = 0f
            velocityY = 0f
            accY = 0f
        }

        if (posY > backgroundHeight - ballSize) {
            posY = backgroundHeight - ballSize
            velocityY = 0f
            accY = 0f
        }
    }

    fun reset() {
        posX = (backgroundWidth - ballSize) / 2f
        posY = (backgroundHeight - ballSize) / 2f
        velocityX = 0f
        velocityY = 0f
        accX = 0f
        accY = 0f
        isFirstUpdate = true
    }
}



