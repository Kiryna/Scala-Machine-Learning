name := "Scala-Machine-Learning"

version := "1.0"

scalaVersion := "2.11.8"

classpathTypes += "maven-plugin"

libraryDependencies ++= Seq(
  "org.nd4j" % "nd4j-native-platform" % "0.7.2",
  "org.nd4j" % "nd4j-api" % "0.7.2",
  "org.nd4j" %% "nd4s" % "0.7.2",
  "org.slf4j" % "slf4j-log4j12" % "1.7.23",
  "junit" % "junit" % "4.10" % "test",
  "org.scalactic" %% "scalactic" % "3.0.1",
  "org.scalatest" %% "scalatest" % "3.0.1" % "test"
  )
    