import org.tensorflow.*;

import java.util.Iterator;

public class ExecuteModelSavedModelBundle {

    public static void main(String[] args) {

        String modelDir = "C:\\Projects\\TensorflowJava\\models\\simple_NN\\SavedModelBuilder";

        SavedModelBundle load = SavedModelBundle.load(modelDir, "serve");
        Graph g = load.graph();
        Session s = load.session();
        float[][] array = new float[1][5];
        for(int i = 0; i<5; i++) {
            array[0][i] = ((i == 0) ? 1 : 0);
            //if (i == 0) array[0][i] = 1;
            //else array[0][i] = 0;
        }
        Tensor<?> data = Tensor.create(array);
        System.out.println(data.shape().length);
        for (Iterator<Operation> it = g.operations(); it.hasNext(); ) {
            Operation op = it.next();
            System.out.println(op.name());
        }
        // Get probabilities
        Tensor result = s.runner().feed("input_placeholder", data).fetch("output_probabilities").run().get(0);
        float[][] resultArray = new float[1][10];
        result.copyTo(resultArray);
        // Get logits
        Tensor resultLogits = s.runner().feed("input_placeholder", data).fetch("logits").run().get(0);
        float[][] logitsArray = new float[1][10];
        resultLogits.copyTo(logitsArray);
        //resultArray = result.copyTo(new float[10][1]);
        load.close();
        //return resultArray;
    }
}
