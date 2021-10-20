import background.Network;
import background.Array;

public class Main {
    public static void main(String[] args) {
        Network net = new Network();
        // System.out.println(net.numLayers);
        // Array.size(net.weight, "weights");
        // Array.size(net.bias, "biases");

        net.SGD(net.weight, 12, 12, 2.00);
    }
}
