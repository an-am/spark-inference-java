package com.spark;

import org.apache.spark.SparkFiles;
import org.bytedeco.opencv.presets.opencv_core;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class TestJnihdf5 {

    private static String MODEL_PATH = "/Users/antonelloamore/IdeaProjects/spark-inference-java/deep_model.h5";

    public static void main(String[] args) throws Exception {

        INDArray input = Nd4j.rand(1, 9);
        System.out.println(input.getFloat(0));
    }
}