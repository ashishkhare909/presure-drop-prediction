// ===============================
// Pressure Drop Prediction Script
// ===============================

// Get form and result elements
const form = document.getElementById("prediction-form");
const resultBox = document.querySelector(".result-content");

// Backend API URL (change later if needed)
const API_URL = "http://127.0.0.1:8000/predict";

// Handle form submission
form.addEventListener("submit", function (event) {
    event.preventDefault(); // Stop page refresh

    // Read input values
    const diameter = document.getElementById("diameter").value;
    const length = document.getElementById("length").value;
    const bendAngle = document.getElementById("bend").value;
    const velocity = document.getElementById("velocity").value;

    // Basic validation
    if (!diameter || !length || !bendAngle || !velocity) {
        alert("Please fill all input fields.");
        return;
    }

    // Prepare data for backend
    const inputData = {
        diameter: parseFloat(diameter),
        length: parseFloat(length),
        bend_angle: parseFloat(bendAngle),
        velocity: parseFloat(velocity)
    };

    // Show loading message
    resultBox.innerHTML = `
        <p><strong>Predicted Pressure Drop:</strong> Calculating...</p>
        <p><strong>Risk Level:</strong> --</p>
        <p><strong>System Message:</strong> Processing input data...</p>
    `;

    // Send data to backend
    fetch(API_URL, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(inputData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error("Server error");
        }
        return response.json();
    })
    .then(data => {
        // Display results
        resultBox.innerHTML = `
            <p><strong>Predicted Pressure Drop:</strong> ${data.pressure_drop} Pa</p>
            <p><strong>Risk Level:</strong> ${data.risk}</p>
            <p><strong>System Message:</strong> ${data.message}</p>
        `;
    })
    .catch(error => {
        // Error handling
        resultBox.innerHTML = `
            <p><strong>Error:</strong> Unable to get prediction.</p>
            <p>Please check backend connection.</p>
        `;
        console.error("Error:", error);
    });
});
