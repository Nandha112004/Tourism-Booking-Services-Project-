from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import pandas as pd
from flask import Flask, render_template, request
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
import re  # For email and username validation

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'static/Database_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tourism.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False, unique=True)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Package Model
class Package(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    package_name = db.Column(db.String(255), nullable=False, unique=True)
    destinations = db.Column(db.String(255), nullable=False)
    package_description = db.Column(db.Text, nullable=False)
    duration = db.Column(db.String(100), nullable=False)
    travel_dates = db.Column(db.String(100), nullable=False)
    price = db.Column(db.Float, nullable=False)
    package_image = db.Column(db.String(255), nullable=True)

class Application(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    package_id = db.Column(db.Integer, db.ForeignKey('package.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    mobile_number = db.Column(db.String(15), nullable=False)
    email_address = db.Column(db.String(120), nullable=False)
    country_of_residence = db.Column(db.String(50), nullable=False)
    address = db.Column(db.Text, nullable=False)
    payment_method = db.Column(db.String(50), nullable=False)
    departure_date = db.Column(db.Date, nullable=False)
    return_date = db.Column(db.Date, nullable=False)
    price_amount = db.Column(db.Float, nullable=False)

    user = db.relationship('User', backref='applications', lazy=True)
    package = db.relationship('Package', backref='applications', lazy=True)

with app.app_context():
    db.create_all()


# Load and clean the data for prediction
file_path = 'API_ST.INT.ARVL_DS2_en_csv_v2_2493.csv'
tourism_data = pd.read_csv(file_path, skiprows=4)

# Data cleaning
tourism_data = tourism_data.drop(columns=['Country Code', 'Indicator Name', 'Indicator Code', 'Unnamed: 68'])
tourism_data = tourism_data.melt(id_vars=['Country Name'], var_name='Year', value_name='Arrivals')
tourism_data = tourism_data.dropna(subset=['Arrivals'])
tourism_data['Year'] = tourism_data['Year'].astype(int)

# Generate a list of available years and future years (Next 10 years)
current_year = datetime.now().year
years = list(sorted(tourism_data['Year'].unique())) + list(range(current_year, current_year + 11))
countries = sorted(tourism_data['Country Name'].unique())

# Function to predict future data using Linear Regression
def predict_future_data(year):
    predictions = []
    for country in countries:
        country_data = tourism_data[tourism_data['Country Name'] == country]
        X = country_data['Year'].values.reshape(-1, 1)
        y = country_data['Arrivals'].values

        if len(X) > 1:
            model = LinearRegression()
            model.fit(X, y)
            predicted_arrival = model.predict(np.array([[year]]))[0]
            predicted_arrival = max(predicted_arrival, 0)  # Ensure no negative arrivals
        else:
            predicted_arrival = np.mean(y) if len(y) > 0 else 0  # Avoid NaN issues

        predictions.append([country, year, predicted_arrival])

    future_data = pd.DataFrame(predictions, columns=['Country Name', 'Year', 'Arrivals'])
    return future_data

# Home page route
@app.route('/')
def home():
    return render_template('home.html')

# Prediction page route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    most_visited = pd.DataFrame(columns=['Country Name', 'Arrivals'])
    least_visited = pd.DataFrame(columns=['Country Name', 'Arrivals'])
    selected_year = None
    country_prediction = None
    selected_country = None

    if request.method == 'POST':
        # Handle year-based prediction
        if 'year' in request.form:
            try:
                selected_year = int(request.form['year'])
                if selected_year in tourism_data['Year'].unique():
                    year_data = tourism_data[tourism_data['Year'] == selected_year]
                else:
                    year_data = predict_future_data(selected_year)

                if not year_data.empty:
                    most_visited = year_data.nlargest(10, 'Arrivals')[['Country Name', 'Arrivals']]
                    least_visited = year_data.nsmallest(10, 'Arrivals')[['Country Name', 'Arrivals']]
            except ValueError:
                print("Invalid year selected")

        # Handle country and year prediction
        if 'country' in request.form and 'country_year' in request.form:
            try:
                selected_country = request.form['country']
                selected_year = int(request.form['country_year'])
                country_data = tourism_data[tourism_data['Country Name'] == selected_country]

                if selected_year in country_data['Year'].unique():
                    country_prediction = country_data[country_data['Year'] == selected_year].iloc[0]['Arrivals']
                else:
                    X = country_data['Year'].values.reshape(-1, 1)
                    y = country_data['Arrivals'].values
                    if len(X) > 1:
                        model = LinearRegression()
                        model.fit(X, y)
                        country_prediction = model.predict(np.array([[selected_year]]))[0]
                        country_prediction = max(country_prediction, 0)
            except ValueError:
                print("Invalid country or year selection")

    return render_template('predict.html',
                           years=years,
                           countries=countries,
                           selected_year=selected_year,
                           most_visited=most_visited,
                           least_visited=least_visited,
                           selected_country=selected_country,
                           country_prediction=country_prediction)


# Signup Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip()
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # ✅ 1. Validate Passwords Match
        if password != confirm_password:
            flash("Passwords do not match!", "error")
            return render_template('signup.html')

        # ✅ 2. Validate Password Strength (Minimum 6 Characters)
        if len(password) < 6:
            flash("Password must be at least 6 characters long!", "error")
            return render_template('signup.html')

        # ✅ 3. Validate Username (No Special Characters, 3-20 Characters)
        if not re.match(r"^[a-zA-Z0-9_]{3,20}$", username):
            flash("Username must be 3-20 characters long and contain only letters, numbers, and underscores.", "error")
            return render_template('signup.html')

        # ✅ 4. Validate Email Format
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            flash("Invalid email address!", "error")
            return render_template('signup.html')

        # ✅ 5. Check if Username or Email Already Exists
        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing_user:
            flash("Username or email already exists!", "error")
            return render_template('signup.html')

        # ✅ 6. Create New User
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, email=email, password=hashed_password)

        try:
            db.session.add(new_user)
            db.session.commit()
            flash("Signup successful! Please login.", "success")
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash("An error occurred while creating your account. Please try again.", "error")

    return render_template('signup.html')



# 🚪 Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # 🛠 Admin Login Check
        if username == "admin" and password == "admin123":
            session['username'] = "admin"
            flash("Admin login successful!", "success")
            return redirect(url_for('admin_dashboard'))

        # 🔍 User Login Check
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash("Login successful! Welcome back, " + username + "!", "success")
            return redirect(url_for('packages'))

        # ❌ Error Message for Incorrect Login
        flash("Incorrect username or password!", "error")
        return redirect(url_for('login'))

    return render_template('login.html')



# Admin Dashboard
@app.route('/admin_dashboard', methods=['GET', 'POST'])
def admin_dashboard():
    # Admin authorization check
    if 'username' not in session or session['username'] != "admin":
        flash("Unauthorized access!", "danger")
        return redirect(url_for('login'))

    if request.method == "POST":
        try:
            # Get form data safely with default values
            package_name = request.form.get('packageName')  # Using .get() avoids KeyError
            destinations = request.form.get('destinations')
            description = request.form.get('package_description')
            duration = request.form.get('duration')
            travel_dates = request.form.get('travel_dates')
            price = request.form.get('price')

            # Handle image upload safely
            image_file = request.files.get("image")
            image_filename = "no_image.png"  # Default if no image is uploaded

            if image_file and allowed_file(image_file.filename):
                image_filename = secure_filename(image_file.filename)
                image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)
                image_file.save(image_path)

            # Store package in the database (SQLAlchemy)
            new_package = Package(
                package_name=package_name,
                destinations=destinations,
                package_description=description,
                duration=duration,
                travel_dates=travel_dates,
                price=price,
                package_image=image_filename
            )
            db.session.add(new_package)
            db.session.commit()

            flash("Package added successfully!", "success")
            return redirect(url_for("admin_dashboard"))

        except Exception as e:
            flash(f"Error: {str(e)}", "error")
            return redirect(url_for("admin_dashboard"))

    # Fetch counts and lists from the database
    user_count = User.query.count()
    user_list = User.query.all()

    package_count = Package.query.count()
    package_list = Package.query.all()

    application_count = Application.query.count()
    application_list = Application.query.all()

    return render_template(
        'admin_dashboard.html',
        user_count=user_count,
        user_list=user_list,
        package_count=package_count,
        package_list=package_list,
        application_count=application_count,
        application_list=application_list
    )


# Allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}


# Packages View for Users
@app.route('/packages')
def packages():
    if 'user_id' not in session:
        flash("Please login to view packages!", "error")
        return redirect(url_for('login'))

    all_packages = Package.query.all()  # Fetch all packages from the database
    return render_template('packages.html', packages=all_packages)


@app.route('/apply/<int:package_id>', methods=['GET', 'POST'])
def apply(package_id):
    package = Package.query.get(package_id)
    if package is None:
        flash("Package not found!", "error")
        return redirect(url_for('packages'))

    # Retrieve user details from the session (assuming the user is logged in)
    user_id = session.get('user_id')
    user = User.query.get(user_id)

    if request.method == 'POST':
        # Collecting form data
        full_name = request.form['full_name']
        mobile_number = request.form['mobile_number']
        email_address = request.form['email_address']
        country_of_residence = request.form['country_of_residence']
        address = request.form['address']
        payment_method = request.form['payment_method']
        departure_date = request.form['departure_date']
        return_date = request.form['return_date']

        # Save the application to the database
        new_application = Application(
            package_id=package_id,
            user_id=user_id,
            full_name=full_name,
            mobile_number=mobile_number,
            email_address=email_address,
            country_of_residence=country_of_residence,
            address=address,
            payment_method=payment_method,
            departure_date=datetime.strptime(departure_date, '%Y-%m-%d'),
            return_date=datetime.strptime(return_date, '%Y-%m-%d'),
            price_amount=package.price
        )
        db.session.add(new_application)
        db.session.commit()

        flash("Application submitted successfully!", "success")
        return redirect(url_for('packages'))  # Redirect to packages page

    return render_template('apply.html', package=package, user=user)





# Logout
@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
