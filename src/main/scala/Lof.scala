/**
  * Created by jinss on 7/19/17.
  */

import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector}
import org.apache.commons.math3.ml.distance.ChebyshevDistance
import org.apache.commons.math3.ml.distance.EuclideanDistance
import org.apache.commons.math3.ml.distance.CanberraDistance
import org.apache.commons.math3.ml.distance.ManhattanDistance

import scala.reflect.ClassTag

object Lof {
  def localOutlierFactor(MinPts: Int, p: DenseVector,
                         y: DenseMatrix, distanceMeasure: String): Double = {
    val nrow = y.numRows
    val oSetTmp = kDistanceAndNeighborhood(MinPts, p, y, distanceMeasure)
    val pNeighborhood = oSetTmp._2
    val tmpSeq = pNeighborhood.map { i =>
      val part1 = (0 until i).toArray
      val part2 = ((i + 1) until nrow).toArray
      val part3 = part1 ++ part2
      val tmpMatrix = rowSliceOfSparkDenseMatrix(part3, y)
      val oVec = oneRowOfSparkDenseMatrix(i, y)
      localReachDensity(MinPts, oVec, tmpMatrix, distanceMeasure)
    }
    val lrdP = localReachDensity(MinPts, p, y, distanceMeasure)
    val cardinal = pNeighborhood.length
    tmpSeq.sum / lrdP / cardinal.toDouble
  }

  private def localReachDensity(MinPts: Int, p: DenseVector,
                                y: DenseMatrix, distanceMeasure: String): Double = {
    val oSetTmp = kDistanceAndNeighborhood(MinPts, p, y, distanceMeasure)
    val pNeighborhood = oSetTmp._2
    val cardinal = pNeighborhood.length
    val lrd = if (cardinal == 0) 1.0e10 else {
      val tmp1 = pNeighborhood.map(i => reachDist(MinPts, p, i, y, distanceMeasure))
      cardinal.toDouble / (tmp1.sum)
    }
    lrd
  }

  private def reachDist(k: Int, p: DenseVector,
                        o: Int, y: DenseMatrix, distanceMeasure: String): Double = {
    val nrow = y.numRows
    val oVec = oneRowOfSparkDenseMatrix(o, y)
    val kDistVecWithO = distanceVector(oVec, y, distanceMeasure)
    val tmp = dropPosition(o, kDistVecWithO)
    val kDistVecTmp = tmp.map(tuple => tuple._1)
    val kDistVecWithoutO = kDistVecTmp
    val kDistPosTmp = tmp.map(tuple => tuple._2)
    val kDistPositionWithoutO = kDistPosTmp
    val kDistDenVec = new DenseVector(kDistVecWithoutO)
    val resTmp = kDistanceVec(k, kDistDenVec)
    val kDistOfO = resTmp._1
    val distanceBetweenPandO = distanceCompute(p.toArray, oVec.toArray, distanceMeasure)
    val rd = if (kDistOfO > distanceBetweenPandO) kDistOfO else distanceBetweenPandO
    rd
  }

  def rowSliceOfSparkDenseMatrix(rowIndex: Array[Int], m: DenseMatrix): DenseMatrix = {
    val nrow = rowIndex.length
    val ncol = m.numCols
    val tmp = rowIndex.flatMap(i => oneRowOfSparkDenseMatrixArray(i, m))
    val tmpMatrix = new DenseMatrix(ncol, nrow, tmp)
    tmpMatrix.transpose
  }

  def colSliceOfSparkDenseMatrix(colIndex: Array[Int], m: DenseMatrix): DenseMatrix = {
    val ncol = colIndex.length
    val nrow = m.numRows
    val tmp = colIndex.flatMap(i => oneColumnOfSparkDenseMatrixArray(i, m))
    new DenseMatrix(nrow, ncol, tmp)
  }

  def rowDropOfSparkDenseMatrix(rowIndex: Int, m: DenseMatrix): DenseMatrix = {
    val rowIndexArray: Array[Int] = Array(rowIndex)
    rowDropOfSparkDenseMatrix(rowIndexArray, m)
  }

  def colDropOfSparkDenseMatrix(colIndex: Int, m: DenseMatrix): DenseMatrix = {
    val colIndexArray: Array[Int] = Array(colIndex)
    colDropOfSparkDenseMatrix(colIndexArray, m)
  }


  def rowDropOfSparkDenseMatrix(rowIndex: Array[Int], m: DenseMatrix): DenseMatrix = {
    val nrow = m.numRows
    val ncol = m.numCols
    val trueDrop = rowIndex.map(i => if (i >= 0) i else nrow + i)
    val over = trueDrop.filter(_ < nrow)
    require(over.length == rowIndex.length)
    val tmpSet1 = (0 until nrow).toSet
    val tmpSet2 = trueDrop.toSet
    val tmpSet3 = tmpSet1 diff tmpSet2
    val tmpArray = tmpSet3.toArray.sorted
    val nrow2 = tmpArray.length
    val tmp = tmpArray.flatMap(i => oneRowOfSparkDenseMatrixArray(i, m))
    val tmpMatrix = new DenseMatrix(ncol, nrow2, tmp)
    tmpMatrix.transpose
  }

  def colDropOfSparkDenseMatrix(colIndex: Array[Int], m: DenseMatrix): DenseMatrix = {
    val ncol = m.numCols
    val nrow = m.numRows
    val trueDrop = colIndex.map(i => if (i >= 0) i else ncol + i)
    val over = trueDrop.filter(_ < ncol)
    require(over.length == colIndex.length)
    val tmpSet1 = (0 until ncol).toSet
    val tmpSet2 = trueDrop.toSet
    val tmpSet3 = tmpSet1 diff tmpSet2
    val tmpArray = tmpSet3.toArray.sorted
    val ncol2 = tmpArray.length
    val tmp = tmpArray.flatMap(i => oneColumnOfSparkDenseMatrixArray(i, m))
    new DenseMatrix(nrow, ncol2, tmp)
  }

  def oneRowOfSparkDenseMatrixArray(rowIndex: Int, m: DenseMatrix): Array[Double] = {
    val nrow = m.numRows
    val trueRowIndex = if (rowIndex >= 0) rowIndex else nrow + rowIndex
    require(trueRowIndex < nrow)
    val ncol = m.numCols
    val tmp = (0 until ncol).map(i => m(trueRowIndex, i))
    tmp.toArray
  }

  def oneColumnOfSparkDenseMatrixArray(colIndex: Int, m: DenseMatrix): Array[Double] = {
    val ncol = m.numCols
    val trueColumnIndex = if (colIndex >= 0) colIndex else colIndex + ncol
    require(trueColumnIndex < ncol)
    val nrow = m.numRows
    val tmp = (0 until nrow).map(i => m(i, trueColumnIndex))
    tmp.toArray
  }

  def oneRowOfSparkDenseMatrix(rowIndex: Int, m: DenseMatrix): DenseVector = {
    val nrow = m.numRows
    val trueRowIndex = if (rowIndex >= 0) rowIndex else nrow + rowIndex
    require(trueRowIndex < nrow)
    val ncol = m.numCols
    val tmp = (0 until ncol).map(i => m(trueRowIndex, i))
    new DenseVector(tmp.toArray)
  }

  def oneColumnOfSparkDenseMatrix(colIndex: Int, m: DenseMatrix): DenseVector = {
    val ncol = m.numCols
    val trueColumnIndex = if (colIndex >= 0) colIndex else colIndex + ncol
    require(trueColumnIndex < ncol)
    val nrow = m.numRows
    val tmp = (0 until nrow).map(i => m(i, trueColumnIndex))
    new DenseVector(tmp.toArray)
  }


  private def kDistanceVec(k: Int,
                           distVec: DenseVector
                          ): (Double, Array[Int], Array[Double]) = {
    val distanceArray = distVec.toArray
    val orderX = order(distanceArray)
    val xSorted = distanceArray.sorted
    val kDistance = xSorted(k - 1)
    val numOfNeighbor = xSorted.filter(_ <= kDistance).length
    val neighborObj = orderX.take(numOfNeighbor)
    val neighborDistance = xSorted.take(numOfNeighbor)
    (kDistance, neighborObj, neighborDistance)
  }

  def kDistanceAndNeighborhood(k: Int, y: DenseMatrix,
                               distanceMeasure: String
                              ): Array[(Double, Array[Int], Array[Double])] = {
    val dM = distanceMatrix(y, distanceMeasure)
    val nrow = dM.numRows
    val ncol = nrow
    val dMArray = dM.toArray
    val tmpSeq = (0 until nrow).map { i =>
      val tmpArray = dMArray.slice(i * nrow, (i + 1) * nrow)
      val tmpDV = new DenseVector(tmpArray)
      val tmpA2 = dropPosition(i, tmpDV)
      val distTmp = (0 until nrow - 1).map(j => tmpA2(j)._1)
      val distVec = new DenseVector(distTmp.toArray)
      val posTmp = (0 until nrow - 1).map(j => tmpA2(j)._2)
      val positionVec = posTmp.toArray
      val mytuple = kDistanceVec(k, distVec)
      val kdist = mytuple._1
      val posTmp2 = mytuple._2
      val kdistVec = mytuple._3
      val kdistNeiTmp = posTmp2.map(i => posTmp(i))
      val kdistNeighbor = kdistNeiTmp.toArray
      (kdist, kdistNeighbor, kdistVec)
    }
    tmpSeq.toArray
  }

  def kDistanceAndNeighborhood(k: Int, x: DenseVector,
                               y: DenseMatrix,
                               distanceMeasure: String
                              ): (Double, Array[Int], Array[Double]) = {
    val distanceV = distanceVector(x, y, distanceMeasure)
    kDistanceVec(k, distanceV)
  }

  private def distanceVector(x: DenseVector,
                             y: DenseMatrix,
                             distanceMeasure: String
                            ): DenseVector = {
    val tmpDistanceName = distanceMeasure.toLowerCase.trim
    val distance = tmpDistanceName match {
      case "chebyshev" => new ChebyshevDistance()
      case "canberra" => new CanberraDistance()
      case "euclidean" => new EuclideanDistance()
      case "manhattan" => new ManhattanDistance()
      case _ => new EuclideanDistance()
    }
    val nrow = y.numRows
    val xArray = x.toArray
    val yIter = y.rowIter
    val distanceVec = yIter.map { yv =>
      val tmpVector = yv.toArray
      distance.compute(xArray, tmpVector)
    }
    val tmp2 = distanceVec.toArray
    new DenseVector(tmp2)
  }

  def dropPositionArray[T: ClassTag](position: Int, vec: Array[T]): Array[T] = {
    val len = vec.length
    val truePos = if (position >= 0) position else position + len
    val tmpVec: Array[T] = if (truePos >= len || truePos < 0) vec else {
      val part1: Array[T] = vec.slice(0, truePos)
      val part2: Array[T] = vec.slice(truePos + 1, len)
      part1 ++ part2
    }
    tmpVec
  }

  private def dropPosition(position: Int, distanceVec: DenseVector): Array[(Double, Int)] = {
    val t1 = dropPositionArray[Double](position, distanceVec.toArray)
    val t2 = (0 until distanceVec.size)
    val t2A = t2.toArray
    val t3 = dropPositionArray[Int](position, t2A)
    val tmpSeq = (0 until distanceVec.size - 1).map(i => (t1(i), t3(i)))
    tmpSeq.toArray
  }

  private def distanceCompute(x1: Array[Double],
                              x2: Array[Double],
                              distanceMeasure: String): Double = {
    require(x1.length == x2.length && x1.length > 0)
    val tmpDistanceName = distanceMeasure.toLowerCase.trim
    val distance = tmpDistanceName match {
      case "chebyshev" => new ChebyshevDistance()
      case "canberra" => new CanberraDistance()
      case "euclidean" => new EuclideanDistance()
      case "manhattan" => new ManhattanDistance()
      case _ => new EuclideanDistance()
    }
    distance.compute(x1, x2)
  }

  def distanceMatrix(x: DenseMatrix,
                     distanceMeasure: String
                    ): DenseMatrix = {
    val dM = distanceMeasure
    val nrow = x.numRows
    val ncol = x.numCols
    val tmpV = (0 until nrow * nrow).map(i => 0.0).toArray
    for (i <- 0 until nrow) {
      val tmpV2Array = oneRowOfSparkDenseMatrixArray(i, x)
      for (k <- 0 until i) {
        val tmpV3Array = oneRowOfSparkDenseMatrixArray(k, x)
        tmpV(i * nrow + k) = distanceCompute(tmpV2Array, tmpV3Array, dM)
        tmpV(k * nrow + i) = tmpV(i * nrow + k)
      }
    }
    new DenseMatrix(nrow, nrow, tmpV)
  }

  def order(x: Array[Int]): Array[Int] = {
    case class tmpClassJuly19(tmpx: Double, position: Int)
    val length = x.length
    val tmpSeq = (0 until length).map(i => tmpClassJuly19(x(i), i))
    tmpSeq.toArray.sortWith(_.tmpx < _.tmpx).map(y => y.position)
  }

  def order(x: Array[Double]): Array[Int] = {
    case class tmpClassJuly19(tmpx: Double, position: Int)
    val length = x.length
    val tmpSeq = (0 until length).map(i => tmpClassJuly19(x(i), i))
    tmpSeq.toArray.sortWith(_.tmpx < _.tmpx).map(y => y.position)
  }

  def zeroArrayDouble(len: Int): Array[Double] = {
    val tmpV = (0 until len).map(i => 0.0)
    tmpV.toArray
  }

  def zeroArrayInt(len: Int): Array[Int] = {
    val tmpV = (0 until len).map(i => 0)
    tmpV.toArray
  }

}
