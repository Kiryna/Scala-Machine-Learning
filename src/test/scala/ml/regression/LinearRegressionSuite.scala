package ml.regression

import java.io.File

import org.junit.runner.RunWith
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.scalactic.TolerantNumerics
import org.scalatest.FunSuite
import org.scalatest.junit.JUnitRunner
import org.scalactic.TripleEquals._

import scala.io.Source


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
@RunWith(classOf[JUnitRunner])
class LinearRegressionSuite extends FunSuite {
  private val line = Source.fromFile("../../src/test/resources/testdata.txt").getLines()
  private val data = line.map(line => {val p = line.split(","); p(0).toDouble -> p(1).toDouble}).toArray
  private val m = data.length
  private val x = Nd4j.create(data map {case (k, _) => Array(1, k)})
  private val y = data map{case (_, v) => v} asNDArray(m, 1)
  private val alpha = 0.01
  private val linearRegression = new LinearRegression(x, y, alpha)

  implicit val doubleEquality = TolerantNumerics.tolerantDoubleEquality(0.01)


  test("Check cost function") {
    val actual = linearRegression.computeCost(theta = Nd4j.zeros(x.size(1), 1))
    val expected = 32.07
    assert(actual === expected)
  }

  test("Check grad function") {
    val expected = Nd4j.create(Array(-5.8391, -65.329))
    val actual = linearRegression.grads(theta = Nd4j.zeros(x.size(1), 1))
    assert(actual == expected)
  }

  test("Check train function") {
    val expectedTheta = Nd4j.create(Array(-3.630291, 1.166362))
    val actualTheta = linearRegression.train(1500)
    assert(actualTheta == expectedTheta)
  }

  test("Check prediction") {
    assert(linearRegression.predict(Nd4j.create(Array(1, 3.5))) === 0.4519767868)
    assert(linearRegression.predict(Nd4j.create(Array(1, 7.0))) === 4.5342450129)
  }

}
