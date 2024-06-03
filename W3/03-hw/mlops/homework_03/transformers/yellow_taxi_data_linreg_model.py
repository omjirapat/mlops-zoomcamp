from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your transformation logic here
    categorical = ['PULocationID', 'DOLocationID']
    y_train = data['duration']

    # Convert the DataFrame to a list of dictionaries
    train_dict = data[categorical].to_dict(orient='records')
    # Initialize the DictVectorizer
    dv = DictVectorizer(sparse=True)
    # Fit and transform the training data
    X_train = dv.fit_transform(train_dict)
    # initialize LinReg model
    lr = LinearRegression()
    # Train the model on the transformed training data
    lr.fit(X_train, y_train)
    # print(f"Intercept: {lr.intercept_}")

    return lr, dv, lr.intercept_, X_train, y_train


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'