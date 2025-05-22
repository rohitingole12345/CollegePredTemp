import zipfile
import os
import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS  # <-- Add this import

app = Flask(__name__)
CORS(app)  # <-- Add this line to allow all origins

# Step 1: Extract the Saved Components
zip_filename = 'mhtcet_recommendation_model.zip'
if os.path.exists(zip_filename):
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        zipf.extractall()

# Step 2: Define the recommend_colleges Function with Updated Logic
def recommend_colleges(caste, branch, percentile, gender, historical_cutoffs, unique_branches, caste_mapping, fallback_mapping, max_recommendations=10):
    if branch not in unique_branches:
        return f"Branch '{branch}' not found in dataset."
    if caste not in caste_mapping:
        return f"Caste '{caste}' not recognized."
    if gender not in ['Male', 'Female']:
        return "Gender must be 'Male' or 'Female'."
    if not (0 <= percentile <= 100):
        return "Percentile must be between 0 and 100."
    
    base_categories = caste_mapping[caste]
    categories = base_categories.copy()
    if gender == 'Male':
        categories = [cat for cat in base_categories if not cat.startswith('L')]
    elif gender == 'Female':
        categories = base_categories

    # Initialize the list of eligible colleges
    eligible = pd.DataFrame()

    # Step 1: Collect Safe Colleges (cutoff in [percentile - 3, percentile], target 7-8)
    safe = historical_cutoffs[
        (historical_cutoffs['Branch'] == branch) &
        (historical_cutoffs['Category'].isin(categories)) &
        (historical_cutoffs['Historical_Cutoff'] <= percentile) &
        (historical_cutoffs['Historical_Cutoff'] >= percentile - 3)  # Narrow range
    ].copy()
    
    # Step 2: Collect Aspirational Colleges (cutoff in [percentile, percentile + 2], target 2-3)
    aspirational = historical_cutoffs[
        (historical_cutoffs['Branch'] == branch) &
        (historical_cutoffs['Category'].isin(categories)) &
        (historical_cutoffs['Historical_Cutoff'] > percentile) &
        (historical_cutoffs['Historical_Cutoff'] <= percentile + 2)
    ].copy()
    
    # Prioritize up to 7 safe colleges, then add up to 3 aspirational
    safe_limit = min(7, len(safe))  # Take up to 7 safe colleges
    remaining_slots = max_recommendations - safe_limit  # Remaining slots for aspirational
    aspirational_limit = min(remaining_slots, 3, len(aspirational))  # Take up to 3 aspirational colleges
    
    eligible = pd.concat([safe.head(safe_limit), aspirational.head(aspirational_limit)], ignore_index=True)
    
    # Step 3: Use Fallback Branches to Fill Remaining Slots, with Narrow Range
    if len(eligible) < max_recommendations and branch in fallback_mapping:
        fallback_branches = fallback_mapping[branch]
        for fb in fallback_branches:
            # Stop if we already have 10 colleges
            if len(eligible) >= max_recommendations:
                break

            # Collect safe colleges from fallback branch (narrow range)
            safe_fallback = historical_cutoffs[
                (historical_cutoffs['Branch'] == fb) &
                (historical_cutoffs['Category'].isin(categories)) &
                (historical_cutoffs['Historical_Cutoff'] <= percentile) &
                (historical_cutoffs['Historical_Cutoff'] >= percentile - 3) &
                (~historical_cutoffs.index.isin(eligible.index))
            ].copy()
            
            # Collect aspirational colleges from fallback branch
            aspirational_fallback = historical_cutoffs[
                (historical_cutoffs['Branch'] == fb) &
                (historical_cutoffs['Category'].isin(categories)) &
                (historical_cutoffs['Historical_Cutoff'] > percentile) &
                (historical_cutoffs['Historical_Cutoff'] <= percentile + 2) &
                (~historical_cutoffs.index.isin(eligible.index))
            ].copy()
            
            # Calculate how many more colleges we need
            current_count = len(eligible)
            needed = max_recommendations - current_count
            
            # Adjust safe and aspirational limits for fallback
            current_safe_count = len(eligible[eligible['Historical_Cutoff'] <= percentile])
            current_aspirational_count = len(eligible[eligible['Historical_Cutoff'] > percentile])
            
            # Target 7 safe colleges in total
            safe_fallback_limit = min(7 - current_safe_count, len(safe_fallback))
            remaining_after_safe = needed - safe_fallback_limit
            
            # Target 3 aspirational colleges in total
            aspirational_fallback_limit = min(3 - current_aspirational_count, remaining_after_safe, len(aspirational_fallback))
            
            # Add fallback colleges
            eligible = pd.concat([
                eligible,
                safe_fallback.head(safe_fallback_limit),
                aspirational_fallback.head(aspirational_fallback_limit)
            ], ignore_index=True)
    
    if eligible.empty:
        return "No colleges found within the specified cutoff range."
    
    eligible['Years_Used'] = eligible['Year_Count']
    return eligible.sort_values(by='Historical_Cutoff', ascending=False)\
        .head(max_recommendations)[['College', 'Category', 'Historical_Cutoff', 'Years_Used']]\
        .reset_index(drop=True)

# Step 3: Load the Saved Components
with open('model_historical_cutoffs_final.pkl', 'rb') as f:
    historical_cutoffs = pickle.load(f)
with open('model_unique_branches_final.pkl', 'rb') as f:
    unique_branches = pickle.load(f)
with open('model_caste_mapping_final.pkl', 'rb') as f:
    caste_mapping = pickle.load(f)
with open('model_fallback_mapping_final.pkl', 'rb') as f:
    fallback_mapping = pickle.load(f)

# Step 4: Define Routes
@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MHT CET College Predictor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {
            background: linear-gradient(to right, #6ee7b7, #3b82f6);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            font-family: 'Arial', sans-serif;
        }
        .container {
            background: white;
            padding: 2rem;
            border-radius: 1rem;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
            max-width: 800px;
            width: 100%;
        }
        .form-group {
            margin-bottom: 1rem;
        }
        .form-group label {
            font-weight: bold;
            color: #333;
        }
        .form-group input, .form-group select {
            width: 100%;
            padding: 0.5rem;
            border: 1px solid #ccc;
            border-radius: 0.5rem;
            font-size: 1rem;
        }
        .btn {
            background-color: #3b82f6;
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 0.5rem;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .btn:hover {
            background-color: #2563eb;
        }
        #loading {
            display: none;
            color: #3b82f6;
            font-style: italic;
        }
        #error {
            color: red;
            font-weight: bold;
            display: none;
        }
        #results {
            margin-top: 2rem;
            display: none;
        }
        #results table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }
        #results th, #results td {
            padding: 0.75rem;
            border: 1px solid #ddd;
            text-align: center;
        }
        #results th {
            background-color: #3b82f6;
            color: white;
            font-weight: bold;
        }
        #results tr:nth-child(even) {
            background-color: #f2f2f2;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-3xl font-bold text-center mb-6 text-gray-800">MHT CET College Predictor</h1>
        <div class="form-group">
            <label for="percentile">Percentile</label>
            <input type="number" id="percentile" step="0.01" value="95.0" min="0" max="100" required>
        </div>
        <div class="form-group">
            <label for="gender">Gender</label>
            <select id="gender" required>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
            </select>
        </div>
        <div class="form-group">
            <label for="caste">Caste</label>
            <select id="caste" required>
                {% for caste in castes %}
                    <option value="{{ caste }}" {% if caste == 'SC' %}selected{% endif %}>{{ caste }}</option>
                {% endfor %}
            </select>
        </div>
        <div class="form-group">
            <label for="branch">Branch</label>
            <select id="branch" required>
                {% for branch in branches %}
                    <option value="{{ branch }}" {% if branch == default_branch %}selected{% endif %}>{{ branch }}</option>
                {% endfor %}
            </select>
        </div>
        <div class="text-center">
            <button class="btn" onclick="getRecommendations()">Get Recommendations</button>
        </div>
        <div id="loading" class="text-center mt-4">Loading recommendations...</div>
        <div id="error" class="text-center mt-4"></div>
        <div id="results" class="mt-4">
            <h2 class="text-xl font-semibold text-gray-800">Recommendations</h2>
            <table id="results-table">
                <thead>
                    <tr>
                        <th>College</th>
                        <th>Category</th>
                        <th>Historical Cutoff</th>
                        <th>Years Used</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>
    </div>

    <script>
        async function getRecommendations() {
            // Get form inputs
            const percentile = document.getElementById('percentile').value;
            const gender = document.getElementById('gender').value;
            const caste = document.getElementById('caste').value;
            const branch = document.getElementById('branch').value;

            // Validate inputs
            if (!percentile || !gender || !caste || !branch) {
                alert('Please fill in all fields.');
                return;
            }

            // Show loading and hide previous results/error
            document.getElementById('loading').style.display = 'block';
            document.getElementById('error').style.display = 'none';
            document.getElementById('results').style.display = 'none';

            try {
                // Make API request
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ percentile, gender, caste, branch })
                });

                const data = await response.json();

                // Hide loading
                document.getElementById('loading').style.display = 'none';

                // Display results or error
                if (data.error) {
                    document.getElementById('error').innerText = data.error;
                    document.getElementById('error').style.display = 'block';
                } else {
                    const tbody = document.querySelector('#results-table tbody');
                    tbody.innerHTML = ''; // Clear previous results
                    data.forEach(row => {
                        const tr = document.createElement('tr');
                        tr.innerHTML = `
                            <td>${row.College}</td>
                            <td>${row.Category}</td>
                            <td>${row.Historical_Cutoff.toFixed(2)}</td>
                            <td>${row.Years_Used}</td>
                        `;
                        tbody.appendChild(tr);
                    });
                    document.getElementById('results').style.display = 'block';
                }
            } catch (error) {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('error').innerText = 'An error occurred while fetching recommendations.';
                document.getElementById('error').style.display = 'block';
            }
        }
    </script>
</body>
</html>
    ''', castes=caste_mapping.keys(), branches=unique_branches, default_branch='Computer Science and Engineering' if 'Computer Science and Engineering' in unique_branches else unique_branches[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    percentile = float(data['percentile'])
    gender = data['gender']
    caste = data['caste']
    branch = data['branch']
    
    result = recommend_colleges(caste, branch, percentile, gender, historical_cutoffs, unique_branches, caste_mapping, fallback_mapping)
    if isinstance(result, str):
        return jsonify({'error': result})
    return jsonify(result.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)