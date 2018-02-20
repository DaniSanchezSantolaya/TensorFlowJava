import org.tensorflow.*;

import java.util.Arrays;
import java.util.Iterator;

public class ExecuteAttentionModel {

    public static void main(String[] args) {

        String modelDir = "C:\\Projects\\TensorflowJava\\models\\attentionRNN2\\SavedModelBuilder";

        // Load Model
        SavedModelBundle load = SavedModelBundle.load(modelDir, "serve");
        Graph g = load.graph();
        Session s = load.session();

        // Create data sample
        int seqLength = 5;
        int inputDim = 3;
        float[][][] inputArray = new float[1][seqLength][inputDim];
        for(int i = 0; i < seqLength; i++) {
            for(int j = 0; j < inputDim; j++ ) {
                inputArray[0][i][j] = 0;
            }
        }
        inputArray[0][0][1] = 1; // Timestep 1
        inputArray[0][1][0] = 1; // Timestep 2
        inputArray[0][2][1] = 1; // Timestep 3
        inputArray[0][3][1] = 1; // Timestep 4
        inputArray[0][4][1] = 1; // Timestep 5
        Tensor<?> data = Tensor.create(inputArray);


        // Execute and get probabilities
        Tensor result = s.runner().feed("x", data).fetch("predictions").run().get(0);
        float[][] resultArray = new float[1][3];
        result.copyTo(resultArray);
        // Get logits
        Tensor resultAttentionWeights = s.runner().feed("x", data).fetch("attention_weights").run().get(0);
        float[][] attentionWeights = new float[1][5];
        resultAttentionWeights.copyTo(attentionWeights);
        load.close();

        // Show results
        for(int i = 0; i < seqLength; i++) {
            System.out.print("Input at time " + i + ":");
            for (int j = 0; j < inputDim; j++) {
                System.out.print(" " + inputArray[0][i][j]);
            }
            System.out.println(" - Attention: " + attentionWeights[0][i]);
        }
        System.out.println("Predictions: ");
        for(int i = 0; i < 3; i++) {
            System.out.println("Class " + i + ": " + resultArray[0][i]);
        }
    }

}
