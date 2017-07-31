name := "Lof"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies  ++= Seq(
  // Last stable release
  "org.scalanlp" %% "breeze" % "0.13.1",
  "org.apache.spark" %% "spark-core" % "2.1.1",
  "org.apache.spark" %% "spark-streaming" % "2.1.1",
  "org.apache.spark" %% "spark-mllib" % "2.1.1",
  "org.apache.spark" %% "spark-mllib-local" % "2.1.1",
  "org.apache.spark" %% "spark-sql" % "2.1.1",
  "org.scalanlp" %% "breeze-natives" % "0.13.1",
  "org.scalanlp" %% "breeze-viz" % "0.13.1",
  "org.apache.commons" % "commons-math3" % "3.6.1"
  )

resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
        
