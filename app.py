from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# =========================
# DATASET
# =========================
data = {
    'CGPA': [6.5, 7.2, 8.0, 5.8, 9.1, 6.9, 7.5, 8.3],
    'Internships': [1, 0, 2, 0, 3, 1, 2, 2],
    'Projects': [2, 1, 3, 1, 4, 2, 3, 3],
    'Communication': [6, 5, 8, 4, 9, 6, 7, 8],
    'Aptitude': [70, 65, 85, 50, 90, 75, 80, 88],
    'Placed': [0, 0, 1, 0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

X = df.drop('Placed', axis=1)
y = df['Placed']

model = LogisticRegression()
model.fit(X, y)

# =========================
# HOME PAGE
# =========================
@app.route('/', methods=['GET', 'POST'])
def home():

    result = ""
    probability = ""
    suggestions = []

    if request.method == 'POST':

        cgpa = float(request.form['cgpa'])
        internships = int(request.form['internships'])
        projects = int(request.form['projects'])
        communication = int(request.form['communication'])
        aptitude = int(request.form['aptitude'])

        student = [[
            cgpa,
            internships,
            projects,
            communication,
            aptitude
        ]]

        prediction = model.predict(student)

        prob = model.predict_proba(student)[0][1] * 100

        probability = f"{prob:.2f}%"

        # Suggestions
        if cgpa < 7:
            suggestions.append(
                "Improve CGPA and academic consistency"
            )

        if internships == 0:
            suggestions.append(
                "Do internships for practical exposure"
            )

        if projects < 2:
            suggestions.append(
                "Build more technical projects"
            )

        if communication < 6:
            suggestions.append(
                "Improve communication skills"
            )

        if aptitude < 70:
            suggestions.append(
                "Practice aptitude and coding problems"
            )

        if prediction[0] == 1:
            result = "Likely to be Placed"
        else:
            result = "Needs Improvement"

    return render_template(
        'index.html',
        result=result,
        probability=probability,
        suggestions=suggestions
    )

# =========================
# RUN
# =========================
if __name__ == '__main__':
    app.run(debug=True)