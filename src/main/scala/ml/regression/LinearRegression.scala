package ml.regression

import ml.linalg.Implicits._
import ml.linalg._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4s.Implicits._
import org.slf4j.LoggerFactory

/*
 * Copyright (C) 2016 Iryna Kharaborkina.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
class LinearRegression(val x: INDArray, val y: INDArray, val alpha: Double, val numIters: Int = 1500) {
  private val logger = LoggerFactory.getLogger(classOf[LinearRegression].getSimpleName)
  private val m = y.size(0)
  private val numFeatures = x.size(1)
  private val initialTheta = (1 to numFeatures).map(_ => 0).asNDArray(numFeatures, 1)
  private val xNoBias = x(**, NDArrayIndex.interval(1, numFeatures))
  private lazy val learnedTheta = train(numIters)

  def predict(features: INDArray, addBias: Boolean = false): Double = sum(features dot learnedTheta)

  private[regression] def train(numitem: Int): INDArray = train(initialTheta, numitem)

  private def train(theta: INDArray, iter: Int): INDArray = {
    if (iter == 0) theta
    else {
      logger.info(s"iteration number #${numIters - iter + 1}")
      logger.info(s"Current cost is ${computeCost(theta)}")
      train(step(theta), iter - 1)
    }
  }

  private[regression] def step(theta: INDArray) = theta - grads(theta) * alpha

  private[regression] def computeCost(theta: INDArray): Double = sum((h(theta) - y) sqr) / (2 * m)

  private[regression] def grads(theta: INDArray): INDArray =  {
    Nd4j.concat(0, sum(h(theta) - y, 1) / m, sum((h(theta) - y) dot xNoBias, 1) / m)
  }

  private[regression] def h(theta: INDArray): INDArray = (x dot theta).transpose()

}




