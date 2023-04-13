# Import necessary libraries
import yfinance as yf
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from bottle import Bottle, run, template

# Create the Bottle app
app = Bottle()

# Define the index route
@app.route('/')
def index():
    # Get live data from Yfinance
    ticker = "EURUSD=X"  # replace with the currency pair of your choice
    data = yf.download(ticker, period="1d", interval="1m")["Close"].reset_index()

    # Prepare data
    X = data.drop(['Close', 'Datetime'], axis=1)  # features
    y = data['Close']  # target variable

    # Split data into training and testing sets
    split = int(0.8 * len(data))  # 80% training, 20% testing
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train the decision tree classifier
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Use the trained model to make predictions on new data
    new_data = yf.download(ticker, period="1d", interval="1m")["Close"].reset_index().drop('Datetime', axis=1)
    new_predictions = model.predict(new_data)

    # Calculate the accuracy of the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Format predictions and accuracy as HTML tags
    prediction = f"<h1>The predicted closing price for {ticker} is {new_predictions[-1]:.2f}</h1>"
    accuracy_tag = f"<h2>The accuracy of the model is {accuracy:.2f}</h2>"

    # Return the predictions in HTML tags
    return prediction,accuracy_tag

# Run the app
if __name__ == '__main__':
    run(app, host='localhost', port=8080, debug=True)
