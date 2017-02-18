package ml.linalg

import org.nd4s.Implicits._
import org.junit.runner.RunWith
import org.scalatest.FunSuite
import org.scalatest.junit.JUnitRunner


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
class PackageObjectSuit extends FunSuite {

  test("Check sum with dimension function") {
    val arr = (1 to 10).asNDArray(5, 2)
    val actualHor = sum(arr, 1)
    val actualVert = sum(arr, 0)
    val expectedHor = Array(3.00, 7.00, 11.00, 15.00, 19.00).asNDArray(5, 1)
    val expectedVert = Array(25, 30).asNDArray(1, 2)
    assert(actualHor == expectedHor)
    assert(actualVert == expectedVert)
  }

}
