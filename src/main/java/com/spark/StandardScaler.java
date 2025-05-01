package com.spark;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.io.File;
import java.util.Map;

public class StandardScaler implements java.io.Serializable {
    private static final long serialVersionUID = 1L;

    private final INDArray mean;
    private final INDArray scale;

    public StandardScaler(INDArray mean, INDArray scale) {
        this.mean = mean;
        this.scale = scale;
    }

    // Applies the standard scaling transformation to a single sample.
    public INDArray transform(INDArray row){
        return row.sub(mean).div(scale);
    }

    // Load scaler parameters from a JSON file.
    public static StandardScaler fromNpyFiles(String scalerPath) throws Exception {
        Map<String,INDArray> arrays = Nd4j.createFromNpzFile(new File(scalerPath));
        return new StandardScaler(arrays.get("mean"), arrays.get("scale"));
    }


}