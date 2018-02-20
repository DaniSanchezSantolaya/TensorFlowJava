import org.tensorflow.*;

import java.util.Iterator;

public class ExecutePythonModel2 {

    public static void main(String[] args) {

        String modelDir = "C:\\Projects\\TensorflowJava\\SaveModelBuilderExample\\models\\1";
        //String modelDir = "C:\\Projects\\TensorflowJava\\models\\simple_NN\\SavedModelBuilder";

        SavedModelBundle load = SavedModelBundle.load(modelDir, "serve");
        Graph g = load.graph();
        Session s = load.session();
        float[][] array = new float[1][784];
        for(int i = 0; i<784; i++) {
            array[0][i] = 0;
        }
        Tensor<?> data = Tensor.create(array);
        System.out.println(data.shape().toString());
        for (Iterator<Operation> it = g.operations(); it.hasNext(); ) {
            Operation op = it.next();
            System.out.println(op.name());
        }
        Tensor result = s.runner().feed("x", data).fetch("y").run().get(0);
        float[][] resultArray = new float[1][10];
        result.copyTo(resultArray);
        System.out.println(resultArray.toString());
        //s.runner().feed("x", data).fetch("y").run();
        //resultArray = result.copyTo(new float[10][1]);
        load.close();
        //return resultArray;
    }

}
