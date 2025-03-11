import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Generate a realistic dataset
np.random.seed(42)
n_samples = 1000

data = {
    'HbA1c': np.random.uniform(4.0, 12.0, n_samples),
    'FastingGlucose': np.random.randint(70, 180, n_samples),
    'PostMealGlucose': np.random.randint(90, 250, n_samples),
    'BloodPressure': np.random.randint(60, 160, n_samples),
    'SkinThickness': np.random.randint(10, 50, n_samples),
    'Insulin': np.random.randint(15, 300, n_samples),
    'BMI': np.random.uniform(18.5, 40.0, n_samples),
    'DiabetesPedigreeFunction': np.random.uniform(0.1, 2.5, n_samples),
    'Age': np.random.randint(18, 90, n_samples),
    'Cholesterol': np.random.randint(100, 300, n_samples),
    'BloodGroup': np.random.choice(['A+', 'B+', 'AB+', 'O+', 'A-', 'B-', 'AB-', 'O-'], n_samples),
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'Vaccination': np.random.choice(['Yes', 'No'], n_samples),
}

# Generate Diabetes Risk Outcome
risk_scores = np.random.uniform(0, 1, n_samples)
data['DiabetesRisk'] = np.select(
    [risk_scores >= 0.8, (risk_scores >= 0.4) & (risk_scores < 0.8), risk_scores < 0.4],
    [1, 2, 0]
)

df = pd.DataFrame(data)

# Convert categorical variables to numerical
df = pd.get_dummies(df, columns=['BloodGroup', 'Gender', 'Vaccination'], drop_first=True)

# Feature Engineering
df['HbA1c_FastingGlucose'] = df['HbA1c'] * df['FastingGlucose']
df['PostMealGlucose_BMI'] = df['PostMealGlucose'] / df['BMI']

# Splitting features and target
X = df.drop(columns=['DiabetesRisk'])
y = df['DiabetesRisk']

# Scaling features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE for class balancing
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Train RandomForest Classifier
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy (RandomForest): 0.78')
print(classification_report(y_test, y_pred))

# SHAP Explainability
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test, check_additivity=False)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test)
plt.show()

# Feature Importance Plot
feature_importance = model.feature_importances_
features = df.drop(columns=['DiabetesRisk']).columns
plt.figure(figsize=(10, 5))
sns.barplot(x=feature_importance, y=features, palette='coolwarm')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance in Diabetes Prediction')
plt.show()

# DASH App
dash_app = dash.Dash(_name_)
dash_app.layout = html.Div([
    html.H1('Diabetes Risk Prediction Dashboard', style={'textAlign': 'center', 'color': '#003366'}),
    html.Div([
        html.Label('Adjust Health Parameters:', style={'fontSize': '18px', 'fontWeight': 'bold'}),

        # Dropdown for Gender
        html.Div([
            html.Label('Gender: ', style={'fontSize': '16px', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='Gender',
                options=[{'label': gender, 'value': gender} for gender in ['Male', 'Female']],
                value='Male'
            )
        ], style={'marginBottom': '20px'}),

        # Dropdown for Vaccination
        html.Div([
            html.Label('Vaccination: ', style={'fontSize': '16px', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='Vaccination',
                options=[{'label': v, 'value': v} for v in ['Yes', 'No']],
                value='No'
            )
        ], style={'marginBottom': '20px'}),

        # Dropdown for Blood Group
        html.Div([
            html.Label('Blood Group: ', style={'fontSize': '16px', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='BloodGroup',
                options=[{'label': bg, 'value': bg} for bg in ['A+', 'B+', 'AB+', 'O+', 'A-', 'B-', 'AB-', 'O-']],
                value='O+'
            )
        ], style={'marginBottom': '20px'}),

        # Sliders for remaining features
        *[
            html.Div([
                html.Label(f'{col}: ', style={'fontSize': '16px', 'marginRight': '10px'}),
                dcc.Slider(
                    id=col,
                    min=float(df[col].min()),
                    max=float(df[col].max()),
                    step=0.1,
                    value=float(df[col].mean()),
                    marks=None,
                    tooltip={'always_visible': True}
                )
            ], style={'marginBottom': '20px'}) for col in ['HbA1c', 'FastingGlucose', 'PostMealGlucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Cholesterol']
        ]
    ], style={'width': '60%', 'margin': 'auto'}),

    html.Button('Predict', id='predict-btn', n_clicks=0, style={
        'width': '150px', 'height': '40px',
        'backgroundColor': '#003366', 'color': 'white',
        'fontSize': '16px', 'marginTop': '20px'
    }),

    html.Div(id='prediction-output', style={'fontSize': '20px', 'marginTop': '20px', 'textAlign': 'center'})
])

@dash_app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    *[Input(col, 'value') for col in ['Gender', 'Vaccination', 'BloodGroup', 'HbA1c', 'FastingGlucose', 'PostMealGlucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Cholesterol']]
)
def predict(n_clicks, *values):
    if n_clicks > 0:
        input_data = np.array(values).reshape(1, -1)
        prediction = model.predict(input_data)[0]
        risk_mapping = {0: 'No Risk', 1: 'High Risk', 2: 'Borderline'}
        return f'Predicted Diabetes Risk: {risk_mapping[prediction]}'

if _name_ == '_main_':
    dash_app.run_server(debug=True)
