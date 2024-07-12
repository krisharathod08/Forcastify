from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import io
from datetime import timedelta

app = Flask(__name__, static_url_path='/static', static_folder='static')
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=10)
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)

class RegistrationForm(FlaskForm):
    username = StringField('username', validators=[DataRequired(), Length(min=2, max=20)])
    password = PasswordField('password', validators=[DataRequired()])
    submit = SubmitField('Sign Up')

def create_user(username, password, is_admin=False):
    new_user = User(username=username, password=password, is_admin=is_admin)
    db.session.add(new_user)
    db.session.commit()

def fetch_all_users():
    return User.query.all()

def fetch_user_by_username(username):
    return User.query.filter_by(username=username).first()

def fetch_user_by_id(user_id):
    return User.query.get(int(user_id))

@app.before_request
def make_session_permanent():
    session.permanent = True
    app.permanent_session_lifetime = timedelta(minutes=10)

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('Preprocess'))
    return redirect(url_for('login'))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = fetch_user_by_username(username)

        if user and user.password == password:
            login_user(user)
            if user.is_admin:
                return redirect(url_for('admin'))
            return redirect(url_for('Preprocess'))
        else:
            flash('Invalid username or password. Please try again.', 'error')

    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegistrationForm()

    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data

        existing_user = fetch_user_by_username(username)
        if existing_user:
            flash('Username already exists. Please choose a different username.', 'error')
        else:
            create_user(username, password)
            flash('Account created successfully. Please log in.', 'success')
            return redirect(url_for('login'))

    return render_template('signup.html', form=form)

@app.route('/Preprocess')
@login_required
def Preprocess():
    return render_template('Preprocess.html')

@app.route('/forecast', methods=['GET', 'POST'])
@login_required
def forecast():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        try:
            df = pd.read_csv(io.StringIO(file.stream.read().decode('utf-8')))
        except pd.errors.EmptyDataError:
            flash('Uploaded file is empty.', 'danger')
            return redirect(request.url)
        df['date'] = pd.to_datetime(df['date'], format='%m-%d-%Y')
        df['value'] = pd.to_numeric(df['value'])
        train_size = int(len(df) * 0.8)
        train, test = df.iloc[:train_size], df.iloc[train_size:]
        selected_algorithm = request.form['algorithm']
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(),
            'Decision Tree': DecisionTreeRegressor(),
            'Support Vector Machine': SVR(),
            'K-Nearest Neighbors': KNeighborsRegressor(),
            'Multi-layer Perceptron': MLPRegressor()
        }
        models[selected_algorithm].fit(train.index.values.reshape(-1, 1), train['value'])
        predictions = models[selected_algorithm].predict(test.index.values.reshape(-1, 1))
        test = test.dropna()
        min_len = min(len(test['value']), len(predictions))
        test_values = test['value'][:min_len]
        predictions = predictions[:min_len]
        mse = np.mean((test_values - predictions) ** 2)
        plt.plot(test.index, test['value'], label='Actual')
        plt.plot(test.index, predictions, label='Predicted')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title(f'{selected_algorithm} Forecast')
        plt.legend()
        plt.savefig('static/forecast_plot.png')
        plt.close()
        return render_template('forecast.html', algorithm=selected_algorithm, mse=mse)
    return render_template('forecast.html')
@app.route('/admin')
@login_required
def admin():
    if not current_user.is_admin:
        flash('You do not have access to this page.', 'error')
        return redirect(url_for('index'))
    users = User.query.all()
    return render_template('admin.html', users=users)

@app.route('/delete_user/<int:user_id>')
@login_required
def delete_user(user_id):
    if not current_user.is_admin:
        flash('You do not have access to this page.', 'error')
        return redirect(url_for('index'))
    user = User.query.get(user_id)
    if user:
        db.session.delete(user)
        db.session.commit()
        flash('User deleted successfully.', 'success')
    return redirect(url_for('admin'))

if __name__ == '__main__':
    app.run(debug=True)
