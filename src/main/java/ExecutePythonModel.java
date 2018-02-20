import org.tensorflow.*;
import org.tensorflow.types.UInt8;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class ExecutePythonModel {

    public static void main(String[] args) {

        String modelDir = "C:\\Projects\\TensorflowJava\\models";
        String imageFile = "C:\\Projects\\TensorflowJava\\data\\mnist\\testSample\\img_1.jpg";

        byte[] graphDef = readAllBytesOrExit(Paths.get(modelDir, "saved_model_2.pb"));
        byte[] imageBytes = readAllBytesOrExit(Paths.get(imageFile));

        float[] array = new float[] { 1, 2, 3, 4, 5 };
        FloatBuffer intBuf = FloatBuffer.wrap(array);
        long[] shape = new long[] {1, 5};
        Tensor<Float> inputTensor = (Tensor<Float>) Tensor.create(array);
        //Tensor.create([5,1], intBuf);
        //Tensor<Float> inputData = Tensor.create(value.getBytes("UTF-8"));
        float[] labelProbabilities = executeTrainedModelGraph(graphDef, inputTensor);
        int bestLabelIdx = maxIndex(labelProbabilities);
        System.out.println(bestLabelIdx);

    }

    private static int maxIndex(float[] probabilities) {
        int best = 0;
        for (int i = 1; i < probabilities.length; ++i) {
            if (probabilities[i] > probabilities[best]) {
                best = i;
            }
        }
        return best;
    }

    private static byte[] readAllBytesOrExit(Path path) {
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(1);
        }
        return null;
    }


    private static Tensor<Float> constructAndExecuteGraphToNormalizeImage(byte[] imageBytes) {

        Graph g = new Graph();
        GraphBuilder b = new GraphBuilder(g);
        // Some constants specific to the pre-trained model at:
        // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
        //
        // - The model was trained with images scaled to 224x224 pixels.
        // - The colors, represented as R, G, B in 1-byte each were converted to
        //   float using (value - Mean)/Scale.
        final int H = 28;
        final int W = 28;
        final float mean = 0f;
        final float scale = 1f;

        // Since the graph is being constructed once per execution here, we can use a constant for the
        // input image. If the graph were to be re-used for multiple input images, a placeholder would
        // have been more appropriate.
        final Output<String> input = b.constant("input", imageBytes);
        final Output<Float> output =
                b.div(
                        b.sub(
                                b.resizeBilinear(
                                        b.expandDims(
                                                b.cast(b.decodeJpeg(input, 3), Float.class),
                                                b.constant("make_batch", 0)),
                                        b.constant("size", new int[] {H, W})),
                                b.constant("mean", mean)),
                        b.constant("scale", scale));

        Session s = new Session(g);
        return s.runner().fetch(output.op().name()).run().get(0).expect(Float.class);

    }


    private static float[] executeTrainedModelGraph(byte[] graphDef, Tensor<Float> inputTensor) {

        Graph g = new Graph();
        g.importGraphDef(graphDef);
        for (Iterator<Operation> it = g.operations(); it.hasNext(); ) {
            Operation op = it.next();
            System.out.println(op.name());
        }
        Session s = new Session(g);
        Tensor<?> result =
                //s.runner().feed("input", inputTensor).fetch("output").run().get(0).expect(Float.class); {
                s.runner().feed("input_placeholder", inputTensor).fetch("logits").run().get(0); {
                    final long[] rshape = result.shape();
                    if (result.numDimensions() != 2 || rshape[0] != 1) {
                        throw new RuntimeException(
                                String.format(
                                        "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                                        Arrays.toString(rshape)));
                    }
                    int nlabels = (int) rshape[1];
                    return result.copyTo(new float[1][nlabels])[0];
            }

    }


    // In the fullness of time, equivalents of the methods of this class should be auto-generated from
    // the OpDefs linked into libtensorflow_jni.so. That would match what is done in other languages
    // like Python, C++ and Go.
    static class GraphBuilder {
        GraphBuilder(Graph g) {
            this.g = g;
        }

        Output<Float> div(Output<Float> x, Output<Float> y) {
            return binaryOp("Div", x, y);
        }

        <T> Output<T> sub(Output<T> x, Output<T> y) {
            return binaryOp("Sub", x, y);
        }

        <T> Output<Float> resizeBilinear(Output<T> images, Output<Integer> size) {
            return binaryOp3("ResizeBilinear", images, size);
        }

        <T> Output<T> expandDims(Output<T> input, Output<Integer> dim) {
            return binaryOp3("ExpandDims", input, dim);
        }

        <T, U> Output<U> cast(Output<T> value, Class<U> type) {
            DataType dtype = DataType.fromClass(type);
            return g.opBuilder("Cast", "Cast")
                    .addInput(value)
                    .setAttr("DstT", dtype)
                    .build()
                    .<U>output(0);
        }

        Output<UInt8> decodeJpeg(Output<String> contents, long channels) {
            return g.opBuilder("DecodeJpeg", "DecodeJpeg")
                    .addInput(contents)
                    .setAttr("channels", channels)
                    .build()
                    .<UInt8>output(0);
        }

        <T> Output<T> constant(String name, Object value, Class<T> type) {
            Tensor<T> t = Tensor.<T>create(value, type);
            return g.opBuilder("Const", name)
                    .setAttr("dtype", DataType.fromClass(type))
                    .setAttr("value", t)
                    .build()
                    .<T>output(0);
        }
        Output<String> constant(String name, byte[] value) {
            return this.constant(name, value, String.class);
        }

        Output<Integer> constant(String name, int value) {
            return this.constant(name, value, Integer.class);
        }

        Output<Integer> constant(String name, int[] value) {
            return this.constant(name, value, Integer.class);
        }

        Output<Float> constant(String name, float value) {
            return this.constant(name, value, Float.class);
        }

        private <T> Output<T> binaryOp(String type, Output<T> in1, Output<T> in2) {
            return g.opBuilder(type, type).addInput(in1).addInput(in2).build().<T>output(0);
        }

        private <T, U, V> Output<T> binaryOp3(String type, Output<U> in1, Output<V> in2) {
            return g.opBuilder(type, type).addInput(in1).addInput(in2).build().<T>output(0);
        }
        private Graph g;
    }

}
