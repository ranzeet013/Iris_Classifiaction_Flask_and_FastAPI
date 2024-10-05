async function makePrediction() {
    const sepal_length = document.getElementById('sepal_length').value;
    const sepal_width = document.getElementById('sepal_width').value;
    const petal_length = document.getElementById('petal_length').value;
    const petal_width = document.getElementById('petal_width').value;

    const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            sepal_length: parseFloat(sepal_length),
            sepal_width: parseFloat(sepal_width),
            petal_length: parseFloat(petal_length),
            petal_width: parseFloat(petal_width)
        })
    });

    const data = await response.json();
    document.getElementById('result').innerText = 
        `Linear Regression: ${data['Linear Regression']}, 
         Logistic Regression: ${data['Logistic Regression']}, 
         SVM: ${data['SVM']}`;
}
