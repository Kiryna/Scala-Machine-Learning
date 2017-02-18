package ml.linalg

import org.nd4j.linalg.api.ndarray.INDArray

class RichArray(iNDArray: INDArray) {
  def sqr = iNDArray mul iNDArray
}
