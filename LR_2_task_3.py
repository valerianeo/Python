from sklearn.datasets import load_iris
iris_dataset = load_iris()

print(f"Iris_dataset keys: \n{iris_dataset.keys()}")
print(iris_dataset['DESCR'][:193] + "\n...")

print(f"Response names: {iris_dataset['target_names']}")
print(f"Feature names: {iris_dataset['feature_names']}")
print(f"Data type: {type(iris_dataset['data'])}")
print(f"Data size: {iris_dataset['data'].shape}")
print(f"The first five lines of data:\n{iris_dataset['data'][:5]}")
print(f"Response array type: {type(iris_dataset['target'])}")
print(f"Size of response array: {iris_dataset['target'].shape}")
print(f"Answers:\n{iris_dataset['target']}")