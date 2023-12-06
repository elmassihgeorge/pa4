from PA4.model import BinaryLogisticRegressionModel


def main():
    blrm = BinaryLogisticRegressionModel(5, 1)
    # Create a dataset for testing
    dataset = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    blrm.train(dataset)
    print("weights", blrm.get_weights())

if __name__ == "__main__":
    main()