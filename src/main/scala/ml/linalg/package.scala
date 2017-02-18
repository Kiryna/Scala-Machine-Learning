package ml

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.indexing.{INDArrayIndex, NDArrayIndex}

import scala.language.implicitConversions

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
package object linalg {

  private[ml] def sum(x: INDArray): Double = x.sumNumber().doubleValue()

  private[ml] def sum(x: INDArray, dim: Int): INDArray = x.sum(dim)

  private[ml] def ** = NDArrayIndex.all()

  object Implicits {
    private[ml] implicit def ind2Rich(indArray: INDArray): RichArray = new RichArray(indArray)

    private[ml] implicit def intToNDArrayIndex(ind: Int): INDArrayIndex = NDArrayIndex.point(ind)
  }
}
