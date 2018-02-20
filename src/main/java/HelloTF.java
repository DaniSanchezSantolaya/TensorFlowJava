
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;

public class HelloTF {
    public static void main(String[] args) throws Exception {

        Graph g = new Graph();
        final  String value = "Hello from " + TensorFlow.version();

        // Construct the computation graph with a single operation, a constant
        // named "MyConst" with a value "value".
        Tensor t = Tensor.create(value.getBytes("UTF-8"));
        g.opBuilder("Const", "MyConst").setAttr("dtype", t.dataType()).setAttr("value", t).build();

        Session s = new Session(g);
        Tensor output = s.runner().fetch("MyConst").run().get(0);
        String strOutput = new String(output.bytesValue(), "UTF-8");
        System.out.println(strOutput);

    }
}
